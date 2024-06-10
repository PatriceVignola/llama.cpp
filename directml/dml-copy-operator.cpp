#include <d3dx12.h>
#include <assert.h>
#include <wil/result.h>
#include "dml-copy-operator.h"

namespace DmlCopy_16_32
{
    #include "generated-shaders/dml-copy-16-32.h"
}

namespace DmlCopy_32_16
{
    #include "generated-shaders/dml-copy-32-16.h"
}

namespace DmlCopy_16_16
{
    #include "generated-shaders/dml-copy-16-16.h"
}

namespace DmlCopy_32_32
{
    #include "generated-shaders/dml-copy-32-32.h"
}

static size_t get_data_type_size(DML_TENSOR_DATA_TYPE dml_datatype) {
    switch (dml_datatype) {
        case DML_TENSOR_DATA_TYPE_FLOAT32: return 4;
        case DML_TENSOR_DATA_TYPE_INT32: return 4;
        case DML_TENSOR_DATA_TYPE_FLOAT16: return 2;
        default: {
            THROW_HR(E_NOTIMPL);
            break;
        }
    }
}

DmlCopyOperator::DmlCopyOperator(
    ID3D12Device* d3d12Device,
    Dml::ExecutionContext* executionContext,
    const dml::TensorDesc& input_tensor_desc,
    const dml::TensorDesc& output_tensor_desc)
        : m_device(d3d12Device)
        , m_executionContext(executionContext)
{
    std::copy_n(input_tensor_desc.sizes.begin(), input_tensor_desc.sizes.size(), m_constants.inputSizes);
    std::copy_n(output_tensor_desc.sizes.begin(), output_tensor_desc.sizes.size(), m_constants.outputSizes);
    std::copy_n(input_tensor_desc.strides->begin(), input_tensor_desc.strides->size(), m_constants.inputStrides);
    std::copy_n(output_tensor_desc.strides->begin(), output_tensor_desc.strides->size(), m_constants.outputStrides);
    m_constants.elementCount = std::accumulate(input_tensor_desc.sizes.begin(), input_tensor_desc.sizes.end(), 1u, std::multiplies<uint32_t>());

    // Compute root signature.
    const int uavCount = 2;
    CD3DX12_ROOT_PARAMETER1 rootParameters[2] = {};

    // Root parameter [0]: UAV descriptor table
    CD3DX12_DESCRIPTOR_RANGE1 descriptorRange(
        D3D12_DESCRIPTOR_RANGE_TYPE_UAV,    // rangeType
        uavCount,                           // numDescriptors
        0,                                  // baseShaderRegister
        0,                                  // registerSpace
        D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
    );
    rootParameters[0].InitAsDescriptorTable(1, &descriptorRange);

    const int constantCount = 18;
    rootParameters[1].InitAsConstants(constantCount, 0);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc(ARRAYSIZE(rootParameters), rootParameters);

    // Create the descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap_desc = {};
    descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    descriptor_heap_desc.NumDescriptors = uavCount;
    descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    THROW_IF_FAILED(d3d12Device->CreateDescriptorHeap(&descriptor_heap_desc, IID_PPV_ARGS(m_heap.ReleaseAndGetAddressOf())));

    ComPtr<ID3DBlob> rootSignatureBlob;
    ComPtr<ID3DBlob> rootSignatureErrorBlob;
    THROW_IF_FAILED(D3D12SerializeVersionedRootSignature(
        &desc,
        rootSignatureBlob.GetAddressOf(),
        rootSignatureErrorBlob.GetAddressOf()
    ));

    THROW_IF_FAILED(m_device->CreateRootSignature(
        0,
        rootSignatureBlob->GetBufferPointer(),
        rootSignatureBlob->GetBufferSize(),
        IID_ID3D12RootSignature,
        &m_rootSignature
    ));

    m_input_dtype_size = get_data_type_size(input_tensor_desc.dataType);
    m_output_dtype_size = get_data_type_size(output_tensor_desc.dataType);

    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
    computePsoDesc.pRootSignature = m_rootSignature.Get();

    if (m_input_dtype_size == 2 && m_output_dtype_size == 4) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlCopy_16_32::g_main, sizeof(DmlCopy_16_32::g_main));
    } else if (m_input_dtype_size == 4 && m_output_dtype_size == 2) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlCopy_32_16::g_main, sizeof(DmlCopy_32_16::g_main));
    } else if (m_input_dtype_size == 2 && m_output_dtype_size == 2) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlCopy_16_16::g_main, sizeof(DmlCopy_16_16::g_main));
    } else if (m_input_dtype_size == 4 && m_output_dtype_size == 4) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlCopy_32_32::g_main, sizeof(DmlCopy_32_32::g_main));
    } else {
        THROW_HR(E_NOTIMPL);
    }

    THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));
}

void DmlCopyOperator::RecordDispatch(
    ID3D12GraphicsCommandList* command_list,
    const Dml::D3D12BufferRegion& temporary_buffer_region)
{
    // Execute the operator
    Dml::DmlCommandRecorder::RecordCustomOperatorDispatch(
        command_list,
        m_rootSignature.Get(),
        m_pipelineState.Get(),
        m_heap.Get(),
        &m_constants,
        m_constants.elementCount,
        sizeof(m_constants) / sizeof(uint32_t)
    );
}

void DmlCopyOperator::UpdateBindings(
    ID3D12Device* d3d12Device,
    void** raw_input_data,
    void* raw_output_data,
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions)
{
    assert(input_buffer_regions.size() == 1);
    assert(output_buffer_regions.size() == 1);

    m_raw_input_data = raw_input_data[0];
    m_raw_output_data = raw_output_data;

    D3D12_UNORDERED_ACCESS_VIEW_DESC input_uav_desc = {};
    input_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    input_uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    input_uav_desc.Buffer.StructureByteStride = static_cast<uint32_t>(m_input_dtype_size);
    input_uav_desc.Buffer.FirstElement = input_buffer_regions[0].Offset() / m_input_dtype_size;
    input_uav_desc.Buffer.NumElements = m_constants.elementCount;

    auto input_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
        m_heap->GetCPUDescriptorHandleForHeapStart(),
        0,
        d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    );

    d3d12Device->CreateUnorderedAccessView(input_buffer_regions[0].GetD3D12Resource(), nullptr, &input_uav_desc, input_cpu_handle);

    D3D12_UNORDERED_ACCESS_VIEW_DESC output_uav_desc = {};
    output_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    output_uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    output_uav_desc.Buffer.StructureByteStride = static_cast<uint32_t>(m_input_dtype_size);
    output_uav_desc.Buffer.FirstElement = output_buffer_regions[0].Offset() / m_output_dtype_size;
    output_uav_desc.Buffer.NumElements = m_constants.elementCount;

    auto output_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
        m_heap->GetCPUDescriptorHandleForHeapStart(),
        1,
        d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    );

    d3d12Device->CreateUnorderedAccessView(output_buffer_regions[0].GetD3D12Resource(), nullptr, &output_uav_desc, output_cpu_handle);
}