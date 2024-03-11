#define NOMINMAX

#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <array>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include "ggml-directml.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#define DML_TARGET_VERSION 0x6300

#include "include/DirectML.h"
#include <wrl/client.h>
#include <dxcore.h>
#include <wil/result.h>
#include <stdexcept>
#include <d3dx12.h>
#include <wil/wrl.h>
#include "DirectMLX.h"
#include "directml/dml-command-recorder.h"
#include "directml/dml-command-queue.h"
#include "directml/dml-pooled-upload-heap.h"
#include "directml/dml-execution-context.h"
#include "directml/dml-allocation-info.h"
#include "directml/dml-reserved-resource-sub-allocator.h"
#include "directml/dml-readback-heap.h"
#include "directml/dml-managed-buffer.h"
#include "directml/dml-copy-operator.h"
#include "directml/dml-operator.h"
#include "directml/dml-graph-operator.h"

using Microsoft::WRL::ComPtr;

void ggml_init_directml() {
    // TODO (pavignol): Implement me
}

static std::string ggml_directml_format_name(int device) {
    return "DirectML" + std::to_string(device);
}

static ComPtr<IDXCoreAdapterList> EnumerateDXCoreAdapters(IDXCoreAdapterFactory* adapter_factory) {
    ComPtr<IDXCoreAdapterList> adapter_list;

    // TODO: use_dxcore_workload_enumeration should be determined by QI
    // When DXCore APIs are available QI for relevant enumeration interfaces
    constexpr bool use_dxcore_workload_enumeration = false;
    if (!use_dxcore_workload_enumeration) {
        // Get a list of all the adapters that support compute
        GUID attributes[]{ DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
        THROW_IF_FAILED(
            adapter_factory->CreateAdapterList(_countof(attributes),
            attributes,
            adapter_list.GetAddressOf()));
    }

    return adapter_list;
}

static void SortDXCoreAdaptersByPreference(IDXCoreAdapterList* adapter_list) {
    if (adapter_list->GetAdapterCount() <= 1) {
        return;
    }

    // DML prefers the HighPerformance adapter by default
    std::array<DXCoreAdapterPreference, 1> adapter_list_preferences = {
        DXCoreAdapterPreference::HighPerformance
    };

  THROW_IF_FAILED(adapter_list->Sort(
    static_cast<uint32_t>(adapter_list_preferences.size()),
    adapter_list_preferences.data()));
}

enum class DeviceType { GPU, NPU, BadDevice };

// Struct for holding each adapter
struct AdapterInfo {
    ComPtr<IDXCoreAdapter> Adapter;
    DeviceType Type; // GPU or NPU
};

static std::vector<AdapterInfo> FilterDXCoreAdapters(IDXCoreAdapterList* adapter_list) {
    auto adapter_infos = std::vector<AdapterInfo>();
    const uint32_t count = adapter_list->GetAdapterCount();
    for (uint32_t i = 0; i < count; ++i) {
        ComPtr<IDXCoreAdapter> candidate_adapter;
        THROW_IF_FAILED(adapter_list->GetAdapter(i, candidate_adapter.GetAddressOf()));

        // Add the adapters that are valid based on the device filter (GPU, NPU, or Both)
        adapter_infos.push_back(AdapterInfo{candidate_adapter, DeviceType::GPU});
    }

    return adapter_infos;
}

static ComPtr<ID3D12Device> ggml_directml_create_d3d12_device() {
    // Create DXCore Adapter Factory
    ComPtr<IDXCoreAdapterFactory> adapter_factory;
    THROW_IF_FAILED(DXCoreCreateAdapterFactory(adapter_factory.GetAddressOf()));

    // Get all DML compatible DXCore adapters
    ComPtr<IDXCoreAdapterList> adapter_list = EnumerateDXCoreAdapters(adapter_factory.Get());

    if (adapter_list->GetAdapterCount() == 0) {
        throw std::runtime_error("No DirectML GPUs or NPUs detected.");
    }

    // Sort the adapter list to honor DXCore hardware ordering
    SortDXCoreAdaptersByPreference(adapter_list.Get());

    // Filter all DXCore adapters to hardware type specified by the device filter
    std::vector<AdapterInfo> adapter_infos = FilterDXCoreAdapters(adapter_list.Get());
    if (adapter_infos.size() == 0) {
        throw std::runtime_error("No devices detected that match the filter criteria.");
    }

    // Create D3D12 Device from DXCore Adapter
    ComPtr<ID3D12Device> d3d12_device;
    THROW_IF_FAILED(D3D12CreateDevice(adapter_infos[0].Adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));

    return d3d12_device;
}

static ComPtr<IDMLDevice> CreateDmlDevice(ID3D12Device* d3d12_device) {
    Microsoft::WRL::ComPtr<IDMLDevice> dml_device;
    THROW_IF_FAILED(DMLCreateDevice1(
        d3d12_device,
        DML_CREATE_DEVICE_FLAG_NONE,
        DML_FEATURE_LEVEL_5_0,
        IID_PPV_ARGS(&dml_device)));

    return dml_device;
}

static D3D12_COMMAND_LIST_TYPE CalculateCommandListType(ID3D12Device* d3d12_device) {
  D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels = {};

  D3D_FEATURE_LEVEL feature_levels_list[] = {
      D3D_FEATURE_LEVEL_1_0_CORE,
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_12_0,
      D3D_FEATURE_LEVEL_12_1
  };

  feature_levels.NumFeatureLevels = ARRAYSIZE(feature_levels_list);
  feature_levels.pFeatureLevelsRequested = feature_levels_list;
  THROW_IF_FAILED(d3d12_device->CheckFeatureSupport(
      D3D12_FEATURE_FEATURE_LEVELS,
      &feature_levels,
      sizeof(feature_levels)
      ));

  auto is_feature_level_1_0_core = (feature_levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE);
  if (is_feature_level_1_0_core) {
    return D3D12_COMMAND_LIST_TYPE_COMPUTE;
  }

  return D3D12_COMMAND_LIST_TYPE_DIRECT;
}

static ComPtr<ID3D12CommandQueue> CreateD3d12CommandQueue(ID3D12Device* d3d12_device) {
    D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {};
    cmd_queue_desc.Type = CalculateCommandListType(d3d12_device);
    cmd_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;

    ComPtr<ID3D12CommandQueue> cmd_queue;
    THROW_IF_FAILED(d3d12_device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(cmd_queue.ReleaseAndGetAddressOf())));

    return cmd_queue;
}

static std::shared_ptr<Dml::DmlGpuAllocator> CreateAllocator(
        ID3D12Device* d3d12_device,
        ID3D12CommandQueue* queue,
        std::shared_ptr<Dml::ExecutionContext> context) {
    auto subAllocator = std::make_shared<Dml::DmlReservedResourceSubAllocator>(
        d3d12_device,
        context,
        queue,
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    auto allocator = std::make_shared<Dml::DmlGpuAllocator>(subAllocator);
    return allocator;
}

// TODO (pavignol): Investigate if we can create a graph plan instead
// A node key uniquely identifies a node/operator based on the sizes, strides, and parameters of their inputs/outputs
struct NodeKey {
    ggml_op op;
    ggml_backend_type backend;
    ggml_type type;
    std::array<int64_t, GGML_MAX_DIMS> ne;
    std::array<int64_t, GGML_MAX_DIMS> nb;
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

    std::vector<std::array<int64_t, GGML_MAX_DIMS>> src_ne;
    std::vector<std::array<int64_t, GGML_MAX_DIMS>> src_nb;
    std::vector<ggml_type> src_types;

    bool operator==(const NodeKey& other) const {
        if (backend != other.backend) return false;
        if (op != other.op) return false;
        if (type != other.type) return false;
        if (ne != other.ne) return false;
        if (nb != other.nb) return false;
        if (memcmp(op_params, other.op_params, GGML_MAX_OP_PARAMS) != 0) return false;
        if (src_ne.size() != other.src_ne.size()) return false;

        for (int i = 0; i < src_ne.size(); ++i) {
            if (src_types[i] != other.src_types[i]) return false;
            if (src_ne != other.src_ne) return false;
            if (src_nb != other.src_nb) return false;
        }

        return true;
    }
};

static NodeKey make_node_key(const ggml_tensor* node) {
    NodeKey node_key;
    node_key.op = node->op;
    node_key.backend = node->backend;
    node_key.type = node->type;
    std::copy(std::begin(node->ne), std::end(node->ne), node_key.ne.begin());
    std::copy(std::begin(node->nb), std::end(node->nb), node_key.nb.begin());
    memcpy(node_key.op_params, node->op_params, GGML_MAX_OP_PARAMS);

    node_key.src_ne.reserve(GGML_MAX_SRC);
    node_key.src_nb.reserve(GGML_MAX_SRC);

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (!node->src[i]) {
            break;
        }

        node_key.src_types.push_back(node->src[i]->type);
        node_key.src_ne.resize(i + 1);
        std::copy(std::begin(node->src[i]->ne), std::end(node->src[i]->ne), node_key.src_ne[i].begin());
        node_key.src_nb.resize(i + 1);
        std::copy(std::begin(node->src[i]->nb), std::end(node->src[i]->nb), node_key.src_nb[i].begin());
    }

    return node_key;
}

struct ggml_directml_context {
    int device;
    ComPtr<ID3D12Device> d3d12_device;
    ComPtr<IDMLDevice> dml_device;
    std::string name;
    ComPtr<ID3D12CommandQueue> d3d12_queue;
    std::shared_ptr<Dml::CommandQueue> command_queue;
    Dml::DmlCommandRecorder command_recorder;
    std::shared_ptr<Dml::ExecutionContext> execution_context;
    Dml::PooledUploadHeap upload_heap;
    ComPtr<ID3D12Fence> fence;
    std::shared_ptr<Dml::DmlGpuAllocator> allocator;
    Dml::ReadbackHeap readback_heap;
    Dml::DmlCommandRecorder* current_recorder = nullptr;

    // TODO (pavignol): Convert to an hash map
    std::vector<std::pair<NodeKey, std::shared_ptr<DmlOperator>>> operator_cache;

    ggml_directml_context(int device)
        : device(device)
        , d3d12_device(ggml_directml_create_d3d12_device())
        , dml_device(CreateDmlDevice(d3d12_device.Get()))
        , name(ggml_directml_format_name(device))
        , d3d12_queue(CreateD3d12CommandQueue(d3d12_device.Get()))
        , command_queue(std::make_shared<Dml::CommandQueue>(d3d12_queue.Get()))
        , command_recorder(d3d12_device.Get(), dml_device.Get(), command_queue)
        , execution_context(std::make_shared<Dml::ExecutionContext>(d3d12_device.Get(), dml_device.Get(), d3d12_queue.Get()))
        , upload_heap(d3d12_device.Get(), execution_context)
        , allocator(CreateAllocator(d3d12_device.Get(), d3d12_queue.Get(), execution_context))
        , readback_heap(d3d12_device.Get()) {
        THROW_IF_FAILED(d3d12_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf())));
        execution_context->SetAllocator(allocator);
    }
};

static ggml_directml_context *s_directml_context = nullptr;

static ggml_guid_t ggml_backend_directml_guid() {
    static ggml_guid guid = { 0x74, 0xad, 0x79, 0x38, 0xc6, 0xc7, 0x4c, 0x99, 0xad, 0x2f, 0x71, 0x9e, 0x80, 0x27, 0x26, 0xcc };
    return &guid;
}

static const char * ggml_backend_directml_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);
    return ctx->name.c_str();
}

struct ggml_backend_directml_buffer_type_context {
    int         device;
    uint64_t    buffer_alignment;
    uint64_t    max_alloc;
    std::string name;

    ggml_backend_directml_buffer_type_context(int device, uint64_t buffer_alignment, uint64_t max_alloc)
        : device(device), buffer_alignment(buffer_alignment), max_alloc(max_alloc), name(ggml_directml_format_name(device)) {}
};

static const char * ggml_backend_directml_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
}

class directml_manager {
public:
    void initialize_device(int device_id) {
        // TODO (pavignol): Implement me
        printf("Initializing Device\n");
    }
};

static directml_manager directmlManager;

struct ggml_directml_memory {
    void* data;
    size_t size = 0;
};

static ggml_directml_memory ggml_directml_allocate(size_t size) {
    ggml_directml_memory memory;
    memory.data = s_directml_context->allocator->Alloc(size);
    memory.size = size;
    return memory;
}

static const char * ggml_backend_directml_buffer_get_name(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buffer->buft->context);
    return ctx->name.c_str();
}

static void ggml_backend_directml_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (s_directml_context) {
        auto * memory = (ggml_directml_memory *)buffer->context;
        s_directml_context->allocator->Free(memory->data);
        delete memory;
    }
}

static void * ggml_backend_directml_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_directml_memory *)buffer->context)->data;
}

static void ggml_backend_directml_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * srcData, size_t offset, size_t size) {
    auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(tensor->data, size);
    ID3D12Resource* dstData = bufferRegion.GetD3D12Resource();

    const auto dstState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
    s_directml_context->upload_heap.BeginUploadToGpu(dstData, bufferRegion.Offset() + offset, dstState, reinterpret_cast<const uint8_t*>(srcData), size);

    // TODO (pavignol): Optimize by not flushing as often
    s_directml_context->execution_context->Flush();
}

static void ggml_backend_directml_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * dstData, size_t offset, size_t size) {
    auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(tensor->data, size);
    ID3D12Resource* srcData = bufferRegion.GetD3D12Resource();

    const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
    // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
    s_directml_context->readback_heap.ReadbackFromGpu(s_directml_context->execution_context.get(), reinterpret_cast<uint8_t*>(dstData), size, srcData, bufferRegion.Offset() + offset, srcState);
}

static void ggml_backend_directml_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // TODO (pavignol): Implement me (set pattern to value)
    printf("ggml_backend_directml_buffer_clear\n");
}

static void ggml_backend_directml_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    tensor->backend = GGML_BACKEND_TYPE_GPU;
}

static ggml_backend_buffer_i ggml_backend_directml_buffer_i = {
    /* .get_name        = */ ggml_backend_directml_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_directml_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_directml_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_directml_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_directml_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_directml_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_directml_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_directml_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (!s_directml_context) {
        s_directml_context = new ggml_directml_context(0);
    }

    auto * ctx = new ggml_directml_memory(ggml_directml_allocate(size));
    return ggml_backend_buffer_init(buft, ggml_backend_directml_buffer_i, ctx, size);
}

static size_t ggml_backend_directml_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buft->context);
    return ctx->buffer_alignment;
}

static size_t ggml_backend_directml_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_directml_buffer_type_context *>(buft->context);
    return ctx->max_alloc;
}

bool ggml_backend_is_directml(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_directml_guid());
}

static bool ggml_backend_directml_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    GGML_UNUSED(buft);
    return ggml_backend_is_directml(backend);
}

static ggml_backend_buffer_type_i ggml_backend_directml_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_directml_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_directml_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_directml_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_directml_buffer_type_get_max_size,
    /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
    /* .supports_backend = */ ggml_backend_directml_buffer_type_supports_backend,
    /* .is_host          = */ NULL,
};


ggml_backend_buffer_type_t ggml_backend_directml_buffer_type(int device) {
    static ggml_backend_buffer_type buffer_type = {
        /* .iface   = */ ggml_backend_directml_buffer_type_interface,
        /* .context = */ new ggml_backend_directml_buffer_type_context(device, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT, UINT32_MAX)
    };

    return &buffer_type;
}

static void ggml_backend_directml_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);

    assert(ctx == s_directml_context);
    s_directml_context = nullptr;
    if (ctx != nullptr) {
        delete ctx;
    }

    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_directml_get_default_buffer_type(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);
    return ggml_backend_directml_buffer_type(ctx->device);
}

static DML_TENSOR_DATA_TYPE ggml_to_dml_datatype(ggml_type ggml_datatype) {
    switch (ggml_datatype) {
        case GGML_TYPE_F32: return DML_TENSOR_DATA_TYPE_FLOAT32;
        case GGML_TYPE_F16: return DML_TENSOR_DATA_TYPE_FLOAT16;
        case GGML_TYPE_I32: return DML_TENSOR_DATA_TYPE_INT32;
        default: {
            THROW_HR(E_NOTIMPL);
            break;
        }
    }
}

static size_t get_data_type_size(DML_TENSOR_DATA_TYPE dml_datatype) {
    switch (dml_datatype) {
        case DML_TENSOR_DATA_TYPE_FLOAT32: return sizeof(float);
        case DML_TENSOR_DATA_TYPE_FLOAT16: return sizeof(ggml_fp16_t);
        case DML_TENSOR_DATA_TYPE_INT32: return sizeof(int32_t);
        default: {
            THROW_HR(E_NOTIMPL);
            break;
        }
    }
}

static dml::TensorDimensions ggml_to_dml_sizes(const ggml_tensor* ggml_tensor) {
    dml::TensorDimensions dml_sizes(GGML_MAX_DIMS);
    for (uint32_t dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        dml_sizes[GGML_MAX_DIMS - dim - 1] = static_cast<uint32_t>(ggml_tensor->ne[dim]);
    }

    return dml_sizes;
}

static dml::TensorStrides ggml_to_dml_strides(const ggml_tensor* ggml_tensor) {
    auto dtype_size = get_data_type_size(ggml_to_dml_datatype(ggml_tensor->type));

    dml::TensorStrides dml_strides(GGML_MAX_DIMS);
    for (uint32_t dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (ggml_tensor->ne[dim] == 1) {
            dml_strides[GGML_MAX_DIMS - dim - 1] = 0;
        } else {
            dml_strides[GGML_MAX_DIMS - dim - 1] = static_cast<uint32_t>(ggml_tensor->nb[dim] / dtype_size);
        }
    }

    return dml_strides;
}

static dml::TensorDesc ggml_to_dml_tensor_desc(const ggml_tensor* ggml_tensor) {
    auto dml_datatype = ggml_to_dml_datatype(ggml_tensor->type);
    auto dml_sizes = ggml_to_dml_sizes(ggml_tensor);
    auto dml_strides = ggml_to_dml_strides(ggml_tensor);
    auto size_in_bytes = DMLCalcBufferTensorSize(dml_datatype, static_cast<uint32_t>(dml_sizes.size()), dml_sizes.data(), dml_strides.data());

    return dml::TensorDesc(dml_datatype, DML_TENSOR_FLAG_NONE, dml_sizes, dml_strides, size_in_bytes, 0);
}

static std::tuple<dml::Expression, dml::Expression> broadcast_tensors(dml::Expression aTensor, dml::Expression bTensor) {
    auto aSizes = aTensor.GetOutputDesc().sizes;
    auto bSizes = bTensor.GetOutputDesc().sizes;
    dml::TensorStrides aStrides(aSizes.size());
    dml::TensorStrides bStrides(bSizes.size());

    uint32_t aStride = 1;
    uint32_t bStride = 1;

    // Broadcast the tensors
    for (int dim_index = static_cast<int>(aTensor.GetOutputDesc().sizes.size() - 1); dim_index >= 0; --dim_index) {
        auto aDim = aTensor.GetOutputDesc().sizes[dim_index];
        auto bDim = bTensor.GetOutputDesc().sizes[dim_index];
        GGML_ASSERT(aDim == bDim || aDim == 1 || bDim == 1);

        if (aDim == 1) {
            aStrides[dim_index] = 0;
            aSizes[dim_index] = bSizes[dim_index];
        } else {
            aStrides[dim_index] = aStride;
            aStride *= aDim;
        }
        
        if (bDim == 1) {
            bStrides[dim_index] = 0;
            bSizes[dim_index] = aSizes[dim_index];
        } else {
            bStrides[dim_index] = bStride;
            bStride *= bDim;
        }
    }

    aTensor = dml::Reinterpret(aTensor, aSizes, aStrides);
    bTensor = dml::Reinterpret(bTensor, bSizes, bStrides);

    return std::make_tuple(aTensor, bTensor);
}

static dml::Expression create_matmul(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 2);

    // The input order in GGML is reversed for MatMul
    auto aTensor = node_inputs[1];
    auto bTensor = node_inputs[0];

    // TODO (pavignol): Instead of doing this, cast all fp32 weights once at the beginning and use only fp16 tensors in the graph
    if (aTensor.GetOutputDesc().dataType != output_tensor_desc.dataType) {
        aTensor = dml::Cast(aTensor, output_tensor_desc.dataType);
    }

    if (bTensor.GetOutputDesc().dataType != output_tensor_desc.dataType) {
        bTensor = dml::Cast(bTensor, output_tensor_desc.dataType);
    }

    auto result = dml::Gemm(aTensor, bTensor, NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE);
    return result;
}

static dml::Expression create_rmsnorm(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc, float epsilon) {
    GGML_ASSERT(node_inputs.size() == 1);

    std::array<uint32_t, 1> axes = {static_cast<uint32_t>(node_inputs[0].GetOutputDesc().sizes.size() - 1)};

    // The input order in GGML is reversed for MatMul
    auto result = dml::MeanVarianceNormalization(node_inputs[0], NullOpt, NullOpt, axes, true, false, epsilon);
    return result;
}

static dml::Expression create_multiply(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 2);

    auto aTensor = node_inputs[0];
    auto bTensor = node_inputs[1];

    std::tie(aTensor, bTensor) = broadcast_tensors(aTensor, bTensor);
    auto result = aTensor * bTensor;
    return result;
}

static std::unique_ptr<DmlCopyOperator> create_copy(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1 || node_inputs.size() == 2);

    return std::make_unique<DmlCopyOperator>(
        s_directml_context->d3d12_device.Get(),
        s_directml_context->execution_context,
        node_inputs[0].GetOutputDesc(),
        output_tensor_desc);
}

static dml::Expression create_add(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 2);

    auto a_tensor = node_inputs[0];
    auto b_tensor = node_inputs[1];
    std::tie(a_tensor, b_tensor) = broadcast_tensors(a_tensor, b_tensor);
    return a_tensor + b_tensor;
}

static dml::Expression create_softmax(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc, float scale, float max_bias) {
    GGML_ASSERT(node_inputs.size() == 1 || node_inputs.size() == 2);
    GGML_ASSERT(max_bias == 0.0f);

    auto input = node_inputs[0];
    std::array<uint32_t, 1> axes = {static_cast<uint32_t>(input.GetOutputDesc().sizes.size() - 1)};

    if (node_inputs.size() == 2) {
        const auto& input_sizes = input.GetOutputDesc().sizes;
        auto mask = node_inputs[1];

        if (mask.GetOutputDesc().dataType != input.GetOutputDesc().dataType) {
            mask = dml::Cast(mask, input.GetOutputDesc().dataType);
        }

        std::tie(input, mask) = broadcast_tensors(input, mask);
        auto result = dml::ActivationSoftmax(input * scale + mask, axes);
        return result;
    }

    if (scale != 1.0f) {
        return dml::ActivationSoftmax(input * scale, axes);
    }

    return dml::ActivationSoftmax(input, axes);
}

static dml::Expression create_relu(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return dml::ActivationRelu(node_inputs[0]);
}

static dml::Expression create_gelu(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return dml::ActivationGelu(node_inputs[0]);
}

static dml::Expression create_silu(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return node_inputs[0] * dml::ActivationSigmoid(node_inputs[0]);
}

static dml::Expression create_abs(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return dml::Abs(node_inputs[0]);
}

static dml::Expression create_neg(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return -node_inputs[0];
}

static dml::Expression create_tanh(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return dml::Tanh(node_inputs[0]);
}

static dml::Expression create_elu(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return dml::ActivationElu(node_inputs[0]);
}

static dml::Expression create_hardsigmoid(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return dml::ActivationHardSigmoid(node_inputs[0], 1.0f/6.0f);
}

static dml::Expression create_hardswish(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return node_inputs[0] * dml::ActivationHardSigmoid(node_inputs[0], 1.0f/6.0f);
}

static DML_SCALAR_UNION generate_scalar_union(DML_TENSOR_DATA_TYPE dtype, double value) {
    DML_SCALAR_UNION scalar{};

    switch (dtype)
    {
    case DML_TENSOR_DATA_TYPE_INT8:
        scalar.Int8 = static_cast<int8_t>(value);
        break;

    case DML_TENSOR_DATA_TYPE_UINT8:
        scalar.UInt8 = static_cast<uint8_t>(value);
        break;

    case DML_TENSOR_DATA_TYPE_INT16:
        scalar.Int16 = static_cast<int16_t>(value);
        break;

    case DML_TENSOR_DATA_TYPE_UINT16:
        scalar.UInt16 = static_cast<uint16_t>(value);
        break;

    case DML_TENSOR_DATA_TYPE_INT32:
        scalar.Int32 = static_cast<int32_t>(value);
        break;

    case DML_TENSOR_DATA_TYPE_UINT32:
        scalar.UInt32 = static_cast<uint32_t>(value);
        break;

    case DML_TENSOR_DATA_TYPE_INT64:
        scalar.Int64 = static_cast<int64_t>(value);
        break;

    case DML_TENSOR_DATA_TYPE_UINT64:
        scalar.UInt64 = static_cast<uint64_t>(value);
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT32:
        scalar.Float32 = static_cast<float>(value);
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT64:
        scalar.Float64 = static_cast<double>(value);
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT16: {
        ggml_fp16_t float16_value = GGML_COMPUTE_FP32_TO_FP16(static_cast<float>(value));
        const BYTE* float16_bytes = reinterpret_cast<const BYTE*>(&float16_value);
        std::copy(float16_bytes, float16_bytes + sizeof(float16_value), scalar.Bytes);
    }
    break;

    default:
        THROW_HR(E_NOTIMPL);
    }

    return scalar;
}

static dml::Expression sequence_tensor(dml::Graph& scope, double start, double step,  DML_TENSOR_DATA_TYPE dtype, const dml::TensorDimensions& sizes)
{
    dml::TensorDesc::Dimensions scalar_dims(sizes.size(), 1);
    dml::TensorDesc::Dimensions scalar_strides(sizes.size(), 0);

    auto start_scalar = generate_scalar_union(dtype, start);
    auto step_scalar = generate_scalar_union(dtype, step);
    auto sequence = dml::FillValueSequence(scope, sizes, dtype, start_scalar, step_scalar);
    return sequence;
}

static dml::Expression scalar_tensor(dml::Graph& scope, double value, DML_TENSOR_DATA_TYPE dtype, const dml::TensorDimensions& sizes)
{
    dml::TensorDesc::Dimensions scalar_dims(sizes.size(), 1);
    dml::TensorDesc::Dimensions scalar_strides(sizes.size(), 0);

    auto scalar = generate_scalar_union(dtype, value);

    auto constant_scalar = dml::Reinterpret(
        dml::FillValueConstant(
            scope,
            scalar_dims,
            dtype,
            scalar),
        sizes,         /* broadcast shape */
        scalar_strides /* broadcast strides */
    );
    return constant_scalar;
}

static dml::Expression create_signum(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return dml::Sign(node_inputs[0]);
}

static dml::Expression create_step(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1);
    return dml::Clip(dml::Sign(node_inputs[0]), 0.0f, 1.0f);
}

static dml::Expression rope_yarn_ramp(dml::Graph& scope, const float low, const float high, dml::Expression pos_range) {
    const auto y = (pos_range / 2 - low) / std::max(0.001f, high - low);
    auto zero = scalar_tensor(scope, 0.0, pos_range.GetOutputDesc().dataType, pos_range.GetOutputDesc().sizes);
    auto one = scalar_tensor(scope, 1.0, pos_range.GetOutputDesc().dataType, pos_range.GetOutputDesc().sizes);
    return 1 - dml::Min(one, dml::Max(zero, y));
}

static std::tuple<dml::Expression, dml::Expression> rope_yarn(
    dml::Graph& scope, dml::Expression theta_extrap, float freq_scale, float corr_dims[2], float ext_factor, float mscale) {
    // Get n-d rotational scaling corrected for extrapolation
    auto theta_interp = freq_scale * theta_extrap;
    auto theta = theta_interp;
    if (ext_factor != 0.0f) {
        auto pos_range = sequence_tensor(scope, 0, 2, DML_TENSOR_DATA_TYPE_FLOAT32, theta_extrap.GetOutputDesc().sizes);
        auto ramp_mix = rope_yarn_ramp(scope, corr_dims[0], corr_dims[1], pos_range) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    auto cos_theta = dml::Cos(theta) * mscale;
    auto sin_theta = dml::Sin(theta) * mscale;

    return std::make_tuple(cos_theta, sin_theta);
}

static std::tuple<dml::Expression, dml::Expression> generate_cos_sin_caches(
    dml::Graph& scope,
    dml::Expression position_ids,
    const dml::TensorDesc& input_desc,
    float freq_scale,
    float corr_dims[2],
    float ext_factor,
    float mscale,
    float theta_scale
) {
    dml::TensorDimensions pos_range_sizes(input_desc.sizes.size(), 1);
    pos_range_sizes.back() = input_desc.sizes.back() / 2;

    auto dml_theta_scale = scalar_tensor(scope, theta_scale, DML_TENSOR_DATA_TYPE_FLOAT32, pos_range_sizes);
    auto exp_range = sequence_tensor(scope, 0, 1, DML_TENSOR_DATA_TYPE_FLOAT32, pos_range_sizes);
    auto exp = dml::Pow(dml_theta_scale, exp_range);
    position_ids = dml::Cast(position_ids, exp.GetOutputDesc().dataType);
    auto theta = dml::Gemm(position_ids, exp, NullOpt, DML_MATRIX_TRANSFORM_TRANSPOSE, DML_MATRIX_TRANSFORM_NONE);

    dml::Expression cos_cache;
    dml::Expression sin_cache;
    std::tie(cos_cache, sin_cache) = rope_yarn(scope, theta, freq_scale, corr_dims, ext_factor, mscale);

    // TODO (pavignol): Check if we still get good results by doing the above calculations in float16 only
    if (cos_cache.GetOutputDesc().dataType != input_desc.dataType) {
        cos_cache = dml::Cast(cos_cache, input_desc.dataType);
        sin_cache = dml::Cast(sin_cache, input_desc.dataType);
    }

    return std::make_tuple(cos_cache, sin_cache);
}

static dml::Expression create_rope(
    dml::Graph& scope,
    const std::vector<dml::Expression>& node_inputs,
    const dml::TensorDesc& output_tensor_desc,
    int mode,
    int n_dims,
    float freq_base,
    float freq_scale,
    int n_orig_ctx,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    float xpos_base,
    bool xpos_down
) {
    // TODO (pavignol): Support all float parameters
    GGML_ASSERT(node_inputs.size() == 2);
    GGML_ASSERT(mode == 0);
    GGML_ASSERT(xpos_base == 0.0f);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, corr_dims);

    auto input = node_inputs[0];
    auto position_ids = node_inputs[1];
    uint32_t batch_size = input.GetOutputDesc().sizes[0];
    uint32_t seq_len = input.GetOutputDesc().sizes[1];
    uint32_t num_heads = input.GetOutputDesc().sizes[2];
    uint32_t head_dim = input.GetOutputDesc().sizes[3];

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    dml::Expression gathered_cos;
    dml::Expression gathered_sin;
    std::tie(gathered_cos, gathered_sin) = generate_cos_sin_caches(scope, position_ids, input.GetOutputDesc(), freq_scale, corr_dims, ext_factor, attn_factor, theta_scale);

    auto reshaped_input = dml::Reinterpret(input, dml::TensorDimensions({batch_size, seq_len, num_heads, head_dim / 2, 2}), NullOpt);

    std::array<uint32_t, 2> split_sizes = {1, 1};
    auto split_input_data = dml::Split(reshaped_input, 4, split_sizes);
    std::swap(split_input_data[0], split_input_data[1]);
    auto rotated_input = dml::Join(split_input_data, 4);

    dml::TensorStrides broadcasted_cos_sin_strides({seq_len * head_dim / 2, head_dim / 2, 0, 1, 0});
    gathered_cos = dml::Reinterpret(gathered_cos, reshaped_input.GetOutputDesc().sizes, broadcasted_cos_sin_strides);
    gathered_sin = dml::Reinterpret(gathered_sin, reshaped_input.GetOutputDesc().sizes, broadcasted_cos_sin_strides);

    auto sign_range = sequence_tensor(scope, -1.0, 2.0, input.GetOutputDesc().dataType, dml::TensorDimensions({2}));
    sign_range = dml::Reinterpret(sign_range, reshaped_input.GetOutputDesc().sizes, dml::TensorStrides({0, 0, 0, 0, 1}));

    auto non_rotated_cos = reshaped_input * gathered_cos;
    auto rotated_sin = rotated_input * gathered_sin * sign_range;
    auto result = non_rotated_cos + rotated_sin;
    result = dml::Reinterpret(result, output_tensor_desc.sizes, output_tensor_desc.strides);
    return result;
}

/*
static std::tuple<dml::Expression, dml::Expression> generate_cos_sin_caches(
    dml::Graph& scope,
    const dml::TensorDesc& input_desc,
    float head_dim,
    float freq_base,
    float max_pos_embeddings
) {
    dml::TensorDimensions pos_range_sizes(input_desc.sizes.size(), 1);
    pos_range_sizes.back() = head_dim / 2;

    DML_SCALAR_UNION pos_range_start{};
    DML_SCALAR_UNION pos_range_delta{};
    pos_range_delta.Float32 = 2.0f / head_dim;
    auto pos_range = dml::FillValueSequence(scope, pos_range_sizes, DML_TENSOR_DATA_TYPE_FLOAT32, pos_range_start, pos_range_delta);

    DML_SCALAR_UNION theta_constant{};
    theta_constant.Float32 = freq_base;
    auto theta = dml::FillValueConstant(scope, dml::TensorDimensions(pos_range_sizes.size(), 1), DML_TENSOR_DATA_TYPE_FLOAT32, theta_constant);
    theta = dml::Reinterpret(theta, pos_range_sizes, dml::TensorStrides(pos_range_sizes.size()));
    auto freqs = 1.0f / dml::Pow(theta, pos_range);

    dml::TensorDimensions indices_sizes(input_desc.sizes.size(), 1);
    indices_sizes.back() = max_pos_embeddings;

    DML_SCALAR_UNION indices_start{};
    DML_SCALAR_UNION indices_delta{};
    indices_delta.Float32 = 1;
    auto indices = dml::FillValueSequence(scope, indices_sizes, DML_TENSOR_DATA_TYPE_FLOAT32, indices_start, indices_delta);

    freqs = dml::Gemm(indices, freqs, NullOpt, DML_MATRIX_TRANSFORM_TRANSPOSE, DML_MATRIX_TRANSFORM_NONE);
    auto cos_cache = dml::Cos(freqs);
    auto sin_cache = dml::Sin(freqs);

    if (cos_cache.GetOutputDesc().dataType != input_desc.dataType) {
        cos_cache = dml::Cast(cos_cache, input_desc.dataType);
        sin_cache = dml::Cast(sin_cache, input_desc.dataType);
    }

    return std::make_tuple(cos_cache, sin_cache);
}

static dml::Expression create_rope(
    dml::Graph& scope,
    const std::vector<dml::Expression>& node_inputs,
    const dml::TensorDesc& output_tensor_desc,
    int mode,
    int n_dims,
    float freq_base,
    float n_orig_ctx,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow
) {
    // TODO (pavignol): Support all float parameters
    GGML_ASSERT(node_inputs.size() == 2);
    GGML_ASSERT(mode == 0);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, corr_dims);

    auto input = node_inputs[0];
    auto position_ids = node_inputs[1];
    uint32_t batch_size = input.GetOutputDesc().sizes[0];
    uint32_t seq_len = input.GetOutputDesc().sizes[1];
    uint32_t num_heads = input.GetOutputDesc().sizes[2];
    uint32_t head_dim = input.GetOutputDesc().sizes[3];

    dml::Expression cos_cache;
    dml::Expression sin_cache;
    std::tie(cos_cache, sin_cache) = generate_cos_sin_caches(scope, input.GetOutputDesc(), head_dim, freq_base, n_orig_ctx);

    auto reshaped_input = dml::Reinterpret(input, dml::TensorDimensions({batch_size, seq_len, num_heads, 2, head_dim / 2}), NullOpt);

    std::array<uint32_t, 2> split_sizes = {1, 1};
    auto split_input_data = dml::Split(reshaped_input, 3, split_sizes);
    std::swap(split_input_data[0], split_input_data[1]);
    auto rotated_input = dml::Join(split_input_data, 3);

    auto gathered_cos = dml::Gather(cos_cache, position_ids, 2, 2);
    auto gathered_sin = dml::Gather(sin_cache, position_ids, 2, 2);

    dml::TensorStrides broadcasted_cos_sin_strides({seq_len * head_dim / 2, head_dim / 2, 0, 0, 1});
    gathered_cos = dml::Reinterpret(gathered_cos, reshaped_input.GetOutputDesc().sizes, broadcasted_cos_sin_strides);
    gathered_sin = dml::Reinterpret(gathered_sin, reshaped_input.GetOutputDesc().sizes, broadcasted_cos_sin_strides);

    DML_SCALAR_UNION sign_range_start{};
    DML_SCALAR_UNION sign_range_delta{};
    if (input.GetOutputDesc().dataType == DML_TENSOR_DATA_TYPE_FLOAT16) {
        const auto valueStart = static_cast<ggml_fp16_t>(-1.0f);
        const auto valueDelta = static_cast<ggml_fp16_t>(2.0f);
        memcpy(sign_range_start.Bytes, reinterpret_cast<const BYTE*>(&valueStart), sizeof(valueStart));
        memcpy(sign_range_delta.Bytes, reinterpret_cast<const BYTE*>(&valueDelta), sizeof(valueDelta));
    } else {
        sign_range_start.Float32 = -1.0f;
        sign_range_delta.Float32 = 2.0f;
    }

    auto sign_range = dml::FillValueSequence(scope, dml::TensorDimensions({2}), input.GetOutputDesc().dataType, sign_range_start, sign_range_delta);
    sign_range = dml::Reinterpret(sign_range, reshaped_input.GetOutputDesc().sizes, dml::TensorStrides({0, 0, 0, 1, 0}));

    auto non_rotated_cos = reshaped_input * gathered_cos;
    auto rotated_sin = rotated_input * gathered_sin * sign_range;
    auto result = non_rotated_cos + rotated_sin;
    result = dml::Reinterpret(result, output_tensor_desc.sizes, output_tensor_desc.strides);
    return result;
}
*/

struct GraphInput {
    GraphInput(dml::Expression dml_tensor, void* raw_data, uint32_t index) : dml_tensor(dml_tensor), raw_data(raw_data), index(index) {}
    const dml::Expression dml_tensor;
    const void* raw_data;
    const uint32_t index;
};

static float fp16_to_fp32(ggml_fp16_t value) {
    return static_cast<float>(GGML_COMPUTE_FP16_TO_FP32(value));
}

static std::shared_ptr<DmlOperator> find_operator(const NodeKey& new_node_key) {
    for (const auto& old_op : s_directml_context->operator_cache) {
        if (new_node_key == old_op.first) {
            return old_op.second;
        }
    }

    return {};
}

static bool ggml_backend_directml_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);

    std::unordered_map<ggml_tensor*, dml::Expression> nodes;
    std::vector<std::vector<ggml_tensor*>> operator_inputs;
    std::vector<std::vector<ggml_tensor*>> operator_outputs;
    std::vector<std::shared_ptr<DmlOperator>> dml_operators;

    for (int node_index = 0; node_index < cgraph->n_nodes; ++node_index) {
        auto node = cgraph->nodes[node_index];
        auto scope = dml::Graph(s_directml_context->dml_device.Get());

        std::vector<dml::TensorDesc> input_tensors;
        std::vector<dml::Expression> node_inputs;
        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            if (!node->src[src_index] || (node->op == GGML_OP_CPY && src_index == 1)) {
                break;
            }

            input_tensors.push_back(ggml_to_dml_tensor_desc(node->src[src_index]));
            node_inputs.push_back(dml::InputTensor(scope, src_index, input_tensors[src_index]));
        }

        auto dml_output_desc = ggml_to_dml_tensor_desc(node);
        bool is_no_op = false;
        scope.PushName(node->name);

        switch (node->op) {
            case GGML_OP_RMS_NORM:
                {
                    auto node_key = make_node_key(node);
                    auto dml_op = find_operator(node_key);

                    if (!dml_op) {
                        float epsilon;
                        memcpy(&epsilon, node->op_params, sizeof(float));
                        auto result = create_rmsnorm(scope, node_inputs, dml_output_desc, epsilon);
                        dml_op = std::make_shared<DmlGraphOperator>(scope, result, s_directml_context->execution_context, *s_directml_context->allocator);
                        s_directml_context->operator_cache.emplace_back(std::move(node_key), dml_op);
                    }

                    dml_operators.push_back(std::move(dml_op));
                }
                break;
            case GGML_OP_MUL_MAT:
                {
                    auto node_key = make_node_key(node);
                    auto dml_op = find_operator(node_key);

                    if (!dml_op) {
                        auto result = create_matmul(scope, node_inputs, dml_output_desc);
                        dml_op = std::make_shared<DmlGraphOperator>(scope, result, s_directml_context->execution_context, *s_directml_context->allocator);
                        s_directml_context->operator_cache.emplace_back(std::move(node_key), dml_op);
                    }

                    dml_operators.push_back(std::move(dml_op));
                }
                break;
            case GGML_OP_MUL:
                {
                    auto node_key = make_node_key(node);
                    auto dml_op = find_operator(node_key);

                    if (!dml_op) {
                        auto result = create_multiply(scope, node_inputs, dml_output_desc);
                        dml_op = std::make_shared<DmlGraphOperator>(scope, result, s_directml_context->execution_context, *s_directml_context->allocator);
                        s_directml_context->operator_cache.emplace_back(std::move(node_key), dml_op);
                    }

                    dml_operators.push_back(std::move(dml_op));
                }
                break;
            case GGML_OP_ROPE:
                {
                    auto node_key = make_node_key(node);
                    auto dml_op = find_operator(node_key);

                    if (!dml_op) {
                        const int n_dims     = ((int32_t *) node->op_params)[1];
                        const int mode       = ((int32_t *) node->op_params)[2];
                        const int n_orig_ctx = ((int32_t *) node->op_params)[4];

                        float xpos_base;
                        bool  xpos_down;

                        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                        memcpy(&freq_base,   (int32_t *) node->op_params +  5, sizeof(float));
                        memcpy(&freq_scale,  (int32_t *) node->op_params +  6, sizeof(float));
                        memcpy(&ext_factor,  (int32_t *) node->op_params +  7, sizeof(float));
                        memcpy(&attn_factor, (int32_t *) node->op_params +  8, sizeof(float));
                        memcpy(&beta_fast,   (int32_t *) node->op_params +  9, sizeof(float));
                        memcpy(&beta_slow,   (int32_t *) node->op_params + 10, sizeof(float));
                        memcpy(&xpos_base,   (int32_t *) node->op_params + 11, sizeof(float));
                        memcpy(&xpos_down,   (int32_t *) node->op_params + 12, sizeof(bool));
                        auto result = create_rope(scope, node_inputs, dml_output_desc, mode, n_dims, freq_base, freq_scale, n_orig_ctx, ext_factor, attn_factor, beta_fast, beta_slow, xpos_base, xpos_down);

                        dml_op = std::make_shared<DmlGraphOperator>(scope, result, s_directml_context->execution_context, *s_directml_context->allocator);
                        s_directml_context->operator_cache.emplace_back(std::move(node_key), dml_op);
                    }

                    dml_operators.push_back(std::move(dml_op));
                }
                break;
            case GGML_OP_CPY:
            case GGML_OP_CONT:
                {
                    auto node_key = make_node_key(node);
                    auto dml_op = find_operator(node_key);

                    if (!dml_op) {
                        dml_op = create_copy(scope, node_inputs, dml_output_desc);
                        s_directml_context->operator_cache.emplace_back(std::move(node_key), dml_op);
                    }

                    dml_operators.push_back(std::move(dml_op));
                }
                break;
            case GGML_OP_SOFT_MAX:
                {
                    auto node_key = make_node_key(node);
                    auto dml_op = find_operator(node_key);

                    if (!dml_op) {
                        float scale;
                        memcpy(&scale, node->op_params, sizeof(float));
                        float max_bias;
                        memcpy(&max_bias, (float *) node->op_params + 1, sizeof(float));
                        auto result = create_softmax(scope, node_inputs, dml_output_desc, scale, max_bias);
                        dml_op = std::make_shared<DmlGraphOperator>(scope, result, s_directml_context->execution_context, *s_directml_context->allocator);
                        s_directml_context->operator_cache.emplace_back(std::move(node_key), dml_op);
                    }

                    dml_operators.push_back(std::move(dml_op));
                }
                break;
            case GGML_OP_ADD:
                {
                    auto node_key = make_node_key(node);
                    auto dml_op = find_operator(node_key);

                    if (!dml_op) {
                        auto result = create_add(scope, node_inputs, dml_output_desc);
                        dml_op = std::make_shared<DmlGraphOperator>(scope, result, s_directml_context->execution_context, *s_directml_context->allocator);
                        s_directml_context->operator_cache.emplace_back(std::move(node_key), dml_op);
                    }

                    dml_operators.push_back(std::move(dml_op));
                }
                break;
            case GGML_OP_UNARY:
                {
                    auto node_key = make_node_key(node);
                    auto dml_op = find_operator(node_key);

                    if (!dml_op) {
                        dml::Expression result;

                        switch (ggml_get_unary_op(node)) {
                        case GGML_UNARY_OP_RELU:
                            result = create_relu(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_GELU_QUICK:
                        case GGML_UNARY_OP_GELU:
                            result = create_gelu(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_SILU:
                            result = create_silu(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_ABS:
                            result = create_abs(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_NEG:
                            result = create_neg(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_TANH:
                            result = create_tanh(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_ELU:
                            result = create_elu(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_HARDSIGMOID:
                            result = create_hardsigmoid(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_SGN:
                            result = create_signum(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_STEP:
                            result = create_step(scope, node_inputs, dml_output_desc);
                            break;
                        case GGML_UNARY_OP_HARDSWISH:
                            result = create_hardswish(scope, node_inputs, dml_output_desc);
                            break;
                        default:
                            THROW_HR(E_NOTIMPL);
                        }

                        dml_op = std::make_shared<DmlGraphOperator>(scope, result, s_directml_context->execution_context, *s_directml_context->allocator);
                        s_directml_context->operator_cache.emplace_back(std::move(node_key), dml_op);
                    }

                    dml_operators.push_back(std::move(dml_op));
                }
                break;
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_NONE:
                // Nothing to do here yet, but we may have to implement it if we move to a graph
                is_no_op = true;
                break;
            default:
                THROW_HR(E_NOTIMPL);
        }

        scope.PopName();

        if (!is_no_op) {
            std::vector<ggml_tensor*> current_inputs;
            for (int src_index = 0; src_index < node_inputs.size(); ++src_index) {
                current_inputs.push_back(node->src[src_index]);
            }

            operator_inputs.push_back(std::move(current_inputs));
            operator_outputs.push_back(std::vector<ggml_tensor*>({node}));
        }
    }

    for (int operator_index = 0; operator_index < dml_operators.size(); ++operator_index) {
        const std::vector<ggml_tensor*> current_operator_inputs = operator_inputs[operator_index];
        const std::vector<ggml_tensor*> current_operator_outputs = operator_outputs[operator_index];

        std::vector<Dml::D3D12BufferRegion> inputBufferRegions;
        inputBufferRegions.reserve(current_operator_inputs.size());
        for (const auto& operator_input : current_operator_inputs) {
            auto input_buffer_region = s_directml_context->allocator->CreateBufferRegion(operator_input->data, ggml_to_dml_tensor_desc(operator_input).totalTensorSizeInBytes);
            inputBufferRegions.push_back(std::move(input_buffer_region));
        }

        std::vector<Dml::D3D12BufferRegion> outputBufferRegions;
        outputBufferRegions.reserve(current_operator_outputs.size());
        for (const auto& operator_output : current_operator_outputs) {
            auto output_buffer_region = s_directml_context->allocator->CreateBufferRegion(operator_output->data, ggml_to_dml_tensor_desc(operator_output).totalTensorSizeInBytes);
            outputBufferRegions.push_back(std::move(output_buffer_region));
        }

        dml_operators[operator_index]->Execute(inputBufferRegions, outputBufferRegions);

/*
        std::vector<std::vector<uint8_t>> srcData(current_operator_inputs.size());
        for (int i = 0; i < current_operator_inputs.size(); ++i) {
            srcData[i] = std::vector<uint8_t>(ggml_to_dml_tensor_desc(current_operator_inputs[i]).totalTensorSizeInBytes);

            auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(current_operator_inputs[i]->data, ggml_to_dml_tensor_desc(current_operator_inputs[i]).totalTensorSizeInBytes);
            ID3D12Resource* srcResource = bufferRegion.GetD3D12Resource();

            const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
            // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
            s_directml_context->readback_heap.ReadbackFromGpu(s_directml_context->execution_context.get(), reinterpret_cast<uint8_t*>(srcData[i].data()), srcData[i].size(), srcResource, bufferRegion.Offset(), srcState);
        }

        {
            auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(current_operator_outputs[0]->data, ggml_to_dml_tensor_desc(current_operator_outputs[0]).totalTensorSizeInBytes);
            ID3D12Resource* srcResource = bufferRegion.GetD3D12Resource();
            std::vector<uint8_t> dstData(ggml_to_dml_tensor_desc(current_operator_outputs[0]).totalTensorSizeInBytes);

            const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
            // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
            s_directml_context->readback_heap.ReadbackFromGpu(s_directml_context->execution_context.get(), reinterpret_cast<uint8_t*>(dstData.data()), dstData.size(), srcResource, bufferRegion.Offset(), srcState);
            // printf("Downloaded!\n");
            // printf("%f\n", fp16_to_fp32(0));
        }

        if (operator_index == dml_operators.size() - 1) {
            auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(current_operator_outputs[0]->data, ggml_to_dml_tensor_desc(current_operator_outputs[0]).totalTensorSizeInBytes);
            ID3D12Resource* srcResource = bufferRegion.GetD3D12Resource();
            std::vector<uint8_t> dstData(ggml_to_dml_tensor_desc(current_operator_outputs[0]).totalTensorSizeInBytes);

            const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
            // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
            s_directml_context->readback_heap.ReadbackFromGpu(s_directml_context->execution_context.get(), reinterpret_cast<uint8_t*>(dstData.data()), dstData.size(), srcResource, bufferRegion.Offset(), srcState);
            // printf("Downloaded!\n");
        }
*/
    }

    if (!dml_operators.empty()) {
        ComPtr<ID3D12Fence> fence;
        uint64_t completion_value;
        s_directml_context->execution_context->ExecuteCommandList(nullptr, fence.GetAddressOf(), &completion_value);
    }

    return true;
}

static bool ggml_directml_supports_op(const struct ggml_tensor * op) {
    switch (op->type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
        break;
    default:
        return false;
    }

    switch (op->op) {
    case GGML_OP_ADD:
    case GGML_OP_MUL:
        // We only support broadcasting when the dimension to broadcast is 1
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (op->src[0]->ne[i] != op->src[1]->ne[i] && op->src[0]->ne[i] != 1 && op->src[1]->ne[i] != 1) {
                return false;
            }
        }
        return true;
    case GGML_OP_MUL_MAT:
        // We only support broadcasting when the dimension to broadcast is 1
        for (int i = GGML_MAX_DIMS - 2; i < GGML_MAX_DIMS; ++i) {
            if (op->src[0]->ne[i] != op->src[1]->ne[i] && op->src[0]->ne[i] != 1 && op->src[1]->ne[i] != 1) {
                return false;
            }
        }
        return true;
    case GGML_OP_SOFT_MAX:
        {
            float max_bias;
            memcpy(&max_bias, (float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
    case GGML_OP_ROPE:
        {
            const int mode = ((int32_t *) op->op_params)[2];
            return mode == 0;
        }
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_SGN:
        case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_HARDSWISH:
            return true;
        default:
            return false;
        }
    case GGML_OP_RMS_NORM:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_PERMUTE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_NONE:
        return true;
    default:
        return false;
    }
}

static bool ggml_backend_directml_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_UNUSED(backend);
    return ggml_directml_supports_op(op);
}

static struct ggml_backend_i directml_backend_i = {
    /* .get_name                = */ ggml_backend_directml_name,
    /* .free                    = */ ggml_backend_directml_free,
    /* .get_default_buffer_type = */ ggml_backend_directml_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_directml_graph_compute,
    /* .supports_op             = */ ggml_backend_directml_supports_op,
};

ggml_backend_t ggml_backend_directml_init(int device) {
    if (!s_directml_context) {
        s_directml_context = new ggml_directml_context(device);
    }

    ggml_backend_t directml_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_directml_guid(),
        /* .interface = */ directml_backend_i,
        /* .context   = */ s_directml_context,
    };

    return directml_backend;
}

static ggml_backend_t ggml_backend_reg_directml_init(const char * params, void * user_data) {
    GGML_UNUSED(params);
    return ggml_backend_directml_init(static_cast<int>(intptr_t(user_data)));
}

extern "C" int ggml_backend_directml_reg_devices();
int ggml_backend_directml_reg_devices() {
    ggml_backend_register(
        ggml_directml_format_name(0).c_str(),
        ggml_backend_reg_directml_init,
        ggml_backend_directml_buffer_type(0),
        reinterpret_cast<void *>(intptr_t(0))
    );

    return 1;
}
