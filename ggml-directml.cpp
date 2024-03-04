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
    auto * memory = (ggml_directml_memory *)buffer->context;
    s_directml_context->allocator->Free(memory->data);
    delete memory;
}

static void * ggml_backend_directml_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_directml_memory *)buffer->context)->data;
}

static void ggml_backend_directml_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * srcData, size_t offset, size_t size) {
    auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(tensor->data, size);
    ID3D12Resource* dstData = bufferRegion.GetD3D12Resource();

    const auto dstState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
    s_directml_context->upload_heap.BeginUploadToGpu(dstData, bufferRegion.Offset(), dstState, reinterpret_cast<const uint8_t*>(srcData), size);

    // TODO (pavignol): Optimize by not flushing as often
    s_directml_context->execution_context->Flush();
}

static void ggml_backend_directml_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * dstData, size_t offset, size_t size) {
    auto bufferRegion = s_directml_context->allocator->CreateBufferRegion(tensor->data, size);
    ID3D12Resource* srcData = bufferRegion.GetD3D12Resource();

    const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state
    // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
    s_directml_context->readback_heap.ReadbackFromGpu(s_directml_context->execution_context.get(), reinterpret_cast<uint8_t*>(dstData), size, srcData, bufferRegion.Offset(), srcState);
}

static void ggml_backend_directml_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // TODO (pavignol): Implement me (set pattern to value)
    printf("ggml_backend_directml_buffer_clear\n");
}

static ggml_backend_buffer_i ggml_backend_directml_buffer_i = {
    /* .get_name        = */ ggml_backend_directml_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_directml_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_directml_buffer_get_base,
    /* .init_tensor     = */ NULL,
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
        /* .context = */ new ggml_backend_directml_buffer_type_context(device, 4, UINT64_MAX)
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
        dml_sizes[GGML_MAX_DIMS - dim - 1] = ggml_tensor->ne[dim];
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
            dml_strides[GGML_MAX_DIMS - dim - 1] = ggml_tensor->nb[dim] / dtype_size;
        }
    }

    return dml_strides;
}

static dml::TensorDesc ggml_to_dml_tensor_desc(const ggml_tensor* ggml_tensor) {
    auto dml_datatype = ggml_to_dml_datatype(ggml_tensor->type);
    auto dml_sizes = ggml_to_dml_sizes(ggml_tensor);
    auto dml_strides = ggml_to_dml_strides(ggml_tensor);
    auto size_in_bytes = DMLCalcBufferTensorSize(dml_datatype, dml_sizes.size(), dml_sizes.data(), dml_strides.data());

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
    for (int dim_index = aTensor.GetOutputDesc().sizes.size() - 1; dim_index >= 0; --dim_index) {
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
    if (aTensor.GetOutputDesc().dataType != bTensor.GetOutputDesc().dataType) {
        if (aTensor.GetOutputDesc().dataType == DML_TENSOR_DATA_TYPE_FLOAT32) {
            aTensor = dml::Cast(aTensor, DML_TENSOR_DATA_TYPE_FLOAT16);
        } else {
            bTensor = dml::Cast(aTensor, DML_TENSOR_DATA_TYPE_FLOAT16);
        }
    }

    auto result = dml::Gemm(aTensor, bTensor, NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE);
    return result;
}

static dml::Expression create_rmsnorm(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc, float epsilon) {
    GGML_ASSERT(node_inputs.size() == 1);

    std::array<uint32_t, 1> axes = {node_inputs[0].GetOutputDesc().sizes.size() - 1};

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

static dml::Expression create_copy(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 1 || node_inputs.size() == 2);

    auto output = node_inputs[0].GetOutputDesc().dataType == output_tensor_desc.dataType
        ? dml::Identity(node_inputs[0])
        : dml::Cast(node_inputs[0], output_tensor_desc.dataType);

    return output;
}

static dml::Expression create_add(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc) {
    GGML_ASSERT(node_inputs.size() == 2);
    return node_inputs[0] + node_inputs[1];
}

static dml::Expression create_softmax(dml::Graph& scope, const std::vector<dml::Expression>& node_inputs, const dml::TensorDesc& output_tensor_desc, float scale) {
    GGML_ASSERT(node_inputs.size() == 2);

    auto input = node_inputs[0];
    auto mask = node_inputs[1];

    if (mask.GetOutputDesc().dataType != input.GetOutputDesc().dataType) {
        mask = dml::Cast(mask, input.GetOutputDesc().dataType);
    }

    std::tie(input, mask) = broadcast_tensors(input, mask);
    std::array<uint32_t, 1> axes = {input.GetOutputDesc().sizes.size() - 1};

    auto result = dml::ActivationSoftmax(input * scale + mask, axes);
    return result;
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
    float freq_base,
    float max_pos_embeddings,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow
) {
    // TODO (pavignol): Support all float parameters
    GGML_ASSERT(node_inputs.size() == 2);

    auto input = node_inputs[0];
    auto position_ids = node_inputs[1];
    uint32_t batch_size = input.GetOutputDesc().sizes[0];
    uint32_t seq_len = input.GetOutputDesc().sizes[1];
    uint32_t num_heads = input.GetOutputDesc().sizes[2];
    uint32_t head_dim = input.GetOutputDesc().sizes[3];

    dml::Expression cos_cache;
    dml::Expression sin_cache;
    std::tie(cos_cache, sin_cache) = generate_cos_sin_caches(scope, input.GetOutputDesc(), head_dim, freq_base, max_pos_embeddings);

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

static struct GraphInput {
    GraphInput(dml::Expression dml_tensor, void* raw_data, uint32_t index) : dml_tensor(dml_tensor), raw_data(raw_data), index(index) {}
    const dml::Expression dml_tensor;
    const void* raw_data;
    const uint32_t index;
};

static bool ggml_backend_directml_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_directml_context *>(backend->context);

    std::unordered_map<ggml_tensor*, dml::Expression> nodes;
    std::vector<dml::Expression> dml_operators;
    std::vector<std::vector<ggml_tensor*>> operator_inputs;
    std::vector<std::vector<ggml_tensor*>> operator_outputs;
    std::vector<dml::Graph> dml_graphs;

    for (int node_index = 0; node_index < cgraph->n_nodes; ++node_index) {
        auto scope = dml::Graph(s_directml_context->dml_device.Get());
        auto node = cgraph->nodes[node_index];

        std::vector<dml::Expression> node_inputs;
        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            if (!node->src[src_index] || (node->op == GGML_OP_CPY && src_index == 1)) {
                break;
            }

            node_inputs.push_back(dml::InputTensor(scope, src_index, ggml_to_dml_tensor_desc(node->src[src_index])));
        }

        dml::Expression result;
        auto dml_output_desc = ggml_to_dml_tensor_desc(node);

        switch (node->op) {
            case GGML_OP_RMS_NORM:
                {
                    float epsilon;
                    memcpy(&epsilon, node->op_params, sizeof(float));
                    result = create_rmsnorm(scope, node_inputs, dml_output_desc, epsilon);
                }
                break;
            case GGML_OP_MUL_MAT:
                result = create_matmul(scope, node_inputs, dml_output_desc);
                break;
            case GGML_OP_MUL:
                result = create_multiply(scope, node_inputs, dml_output_desc);
                break;
            case GGML_OP_ROPE:
                {
                    const int n_dims     = ((int32_t *) node->op_params)[1];
                    const int mode       = ((int32_t *) node->op_params)[2];
                    const int max_pos_embeddings = ((int32_t *) node->op_params)[4];

                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                    memcpy(&freq_base,   (int32_t *) node->op_params +  5, sizeof(float));
                    memcpy(&freq_scale,  (int32_t *) node->op_params +  6, sizeof(float));
                    memcpy(&ext_factor,  (int32_t *) node->op_params +  7, sizeof(float));
                    memcpy(&attn_factor, (int32_t *) node->op_params +  8, sizeof(float));
                    memcpy(&beta_fast,   (int32_t *) node->op_params +  9, sizeof(float));
                    memcpy(&beta_slow,   (int32_t *) node->op_params + 10, sizeof(float));
                    result = create_rope(scope, node_inputs, dml_output_desc, freq_base, max_pos_embeddings, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
                break;
            case GGML_OP_CPY:
            case GGML_OP_CONT:
                result = create_copy(scope, node_inputs, dml_output_desc);
                break;
            case GGML_OP_SOFT_MAX:
                {
                    float scale;
                    memcpy(&scale, node->op_params, sizeof(float));
                    result = create_softmax(scope, node_inputs, dml_output_desc, scale);
                }
                break;
            case GGML_OP_ADD:
                result = create_add(scope, node_inputs, dml_output_desc);
                break;

            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_RELU:
                    result = create_relu(scope, node_inputs, dml_output_desc);
                    break;
                case GGML_UNARY_OP_GELU:
                    result = create_gelu(scope, node_inputs, dml_output_desc);
                    break;
                case GGML_UNARY_OP_SILU:
                    result = create_silu(scope, node_inputs, dml_output_desc);
                    break;
                default:
                    THROW_HR(E_NOTIMPL);
                }
                break;
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
                // Nothing to do here yet, but we may have to implement it if we move to a graph
                break;
            default:
                THROW_HR(E_NOTIMPL);
        }

        if (result) {
            dml_graphs.push_back(std::move(scope));
            dml_operators.push_back(result);

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

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op = dml_graphs[operator_index].Compile(DML_EXECUTION_FLAG_NONE, {dml_operators[operator_index]});
        ComPtr<ID3D12Resource> persistentResource;
        DML_BUFFER_BINDING persistentResourceBinding;
        ComPtr<Dml::DmlManagedBuffer> managedPersistentBuffer;
        DML_BINDING_DESC persistentResourceBindingDesc{};

        uint64_t persistentResourceSize = compiled_op->GetBindingProperties().PersistentResourceSize;
        if (persistentResourceSize > 0)
        {
            auto buffer = s_directml_context->allocator->AllocateDefaultBuffer(persistentResourceSize, Dml::AllocatorRoundingMode::Disabled);
            persistentResource = buffer.GetD3D12Resource();
            persistentResourceBinding = buffer.GetBufferBinding();
            managedPersistentBuffer = wil::MakeOrThrow<Dml::DmlManagedBuffer>(std::move(buffer));
            s_directml_context->execution_context->QueueReference(managedPersistentBuffer.Get());
            persistentResourceBindingDesc = DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &persistentResourceBinding };
        }

        DML_BINDING_DESC initInputBindings{};

        s_directml_context->execution_context->InitializeOperator(
            compiled_op.Get(),
            persistentResourceBindingDesc,
            initInputBindings);

        // Queue references to objects which must be kept alive until resulting GPU work completes
        s_directml_context->execution_context->QueueReference(compiled_op.Get());
        s_directml_context->execution_context->QueueReference(persistentResource.Get());

        auto FillBindingsFromBuffers = [](auto& bufferBindings, auto& bindingDescs, std::vector<Dml::D3D12BufferRegion>& bufferRegions)
        {
            for (auto& bufferRegion : bufferRegions)
            {
                bufferBindings.push_back(bufferRegion.GetBufferBinding());
                bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
            }
        };

        // Bind the inputs
        std::vector<Dml::D3D12BufferRegion> inputBufferRegions;
        inputBufferRegions.reserve(current_operator_inputs.size());
        for (const auto& operator_input : current_operator_inputs) {
            auto input_buffer_region = s_directml_context->allocator->CreateBufferRegion(operator_input->data, ggml_to_dml_tensor_desc(operator_input).totalTensorSizeInBytes);
            inputBufferRegions.push_back(std::move(input_buffer_region));
        }

        std::vector<DML_BUFFER_BINDING> inputBufferBindings;
        inputBufferBindings.reserve(current_operator_inputs.size());
        std::vector<DML_BINDING_DESC> inputBindings;
        inputBindings.reserve(current_operator_inputs.size());
        FillBindingsFromBuffers(inputBufferBindings, inputBindings, inputBufferRegions);

        // Bins the outputs
        std::vector<Dml::D3D12BufferRegion> outputBufferRegions;
        outputBufferRegions.reserve(current_operator_outputs.size());
        for (const auto& operator_output : current_operator_outputs) {
            auto output_buffer_region = s_directml_context->allocator->CreateBufferRegion(operator_output->data, operator_output->nb[GGML_MAX_DIMS - 1]);
            outputBufferRegions.push_back(std::move(output_buffer_region));
        }

        std::vector<DML_BUFFER_BINDING> outputBufferBindings;
        outputBufferBindings.reserve(current_operator_outputs.size());
        std::vector<DML_BINDING_DESC> outputBindings;
        outputBindings.reserve(current_operator_outputs.size());
        FillBindingsFromBuffers(outputBufferBindings, outputBindings, outputBufferRegions);

        s_directml_context->execution_context->ExecuteOperator(compiled_op.Get(), persistentResourceBindingDesc, inputBindings, outputBindings);
    }

    ComPtr<ID3D12Fence> fence;
    uint64_t completion_value;
    s_directml_context->execution_context->ExecuteCommandList(nullptr, fence.GetAddressOf(), &completion_value);

    return true;
}

static bool ggml_directml_supports_op(const struct ggml_tensor * op) {
    // TODO (pavignol): Implement me (look at ggml_vk_supports_op to see how to parse ops)
    return true;
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
    return ggml_backend_directml_init(intptr_t(user_data));
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
