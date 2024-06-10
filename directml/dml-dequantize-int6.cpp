#include <d3dx12.h>
#include <assert.h>
#include <wil/result.h>
#include "dml-dequantize-int6.h"
#include "ggml-quants.h"

namespace DmlDequantizeInt6_16
{
    #include "generated-shaders/dml-dequantize-int6-16.h"
}

namespace DmlDequantizeInt6_32
{
    #include "generated-shaders/dml-dequantize-int6-32.h"
}

DmlDequantizeInt6Operator::DmlDequantizeInt6Operator(
    ID3D12Device* d3d12Device,
    Dml::ExecutionContext* executionContext,
    uint32_t k,
    DML_TENSOR_DATA_TYPE output_data_type)
        : m_device(d3d12Device)
        , m_executionContext(executionContext)
{
    m_groupCount = k / QK_K;

    // Compute root signature.
    const int uavCount = 3;
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

    // TODO (pavignol): Clean me up
    const int constantCount = 0;
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

    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
    computePsoDesc.pRootSignature = m_rootSignature.Get();

    if (output_data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlDequantizeInt6_16::g_main, sizeof(DmlDequantizeInt6_16::g_main));
        m_output_data_type_size = 2;
    } else if (output_data_type == DML_TENSOR_DATA_TYPE_FLOAT32) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlDequantizeInt6_32::g_main, sizeof(DmlDequantizeInt6_32::g_main));
        m_output_data_type_size = 4;
    } else {
        THROW_HR(E_NOTIMPL);
    }

    THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));
}

void DmlDequantizeInt6Operator::RecordDispatch(
    ID3D12GraphicsCommandList* command_list,
    const Dml::D3D12BufferRegion& temporary_buffer_region)
{
    // Execute the operator
    Dml::DmlCommandRecorder::RecordCustomOperatorDispatchByGroup(
        command_list,
        m_rootSignature.Get(),
        m_pipelineState.Get(),
        m_heap.Get(),
        &m_constants,
        0,
        m_groupCount,
        1,
        1);
}

void DmlDequantizeInt6Operator::UpdateBindings(
    ID3D12Device* d3d12Device,
    void** raw_input_data,
    void* raw_output_data,
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions)
{
    assert(input_buffer_regions.size() == 2);
    assert(output_buffer_regions.size() == 1);

    m_raw_input_data[0] = raw_input_data[0];
    m_raw_input_data[1] = raw_input_data[1];
    m_raw_output_data = raw_output_data;

    constexpr size_t block_size_without_scale = sizeof(block_q6_K) - sizeof(ggml_fp16_t);

    // Create the packed block UAV. We use a raw buffer because the block has a weird byte size of 208. Even though it's a multiple of 4,
    // the offset is not necessarily divisible by 208 so we cannot specify a structured buffer here in terms of elements.
    D3D12_UNORDERED_ACCESS_VIEW_DESC block_input_uav_desc = {};
    block_input_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    block_input_uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    block_input_uav_desc.Buffer.FirstElement = input_buffer_regions[0].Offset() / sizeof(uint32_t);
    block_input_uav_desc.Buffer.NumElements = static_cast<uint32_t>(input_buffer_regions[0].SizeInBytes() / sizeof(uint32_t));
    block_input_uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

    auto block_input_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
        m_heap->GetCPUDescriptorHandleForHeapStart(),
        0,
        d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    );

    d3d12Device->CreateUnorderedAccessView(input_buffer_regions[0].GetD3D12Resource(), nullptr, &block_input_uav_desc, block_input_cpu_handle);

    // Create the super-block scale UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC scale_input_uav_desc = {};
    scale_input_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    scale_input_uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    scale_input_uav_desc.Buffer.StructureByteStride = sizeof(ggml_fp16_t);
    scale_input_uav_desc.Buffer.FirstElement = input_buffer_regions[1].Offset() / sizeof(ggml_fp16_t);
    scale_input_uav_desc.Buffer.NumElements = static_cast<uint32_t>(input_buffer_regions[1].SizeInBytes() / sizeof(ggml_fp16_t));

    auto scale_input_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
        m_heap->GetCPUDescriptorHandleForHeapStart(),
        1,
        d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    );

    d3d12Device->CreateUnorderedAccessView(input_buffer_regions[1].GetD3D12Resource(), nullptr, &scale_input_uav_desc, scale_input_cpu_handle);

    // Create the output UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC output_uav_desc = {};
    output_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    output_uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    output_uav_desc.Buffer.StructureByteStride = static_cast<uint32_t>(m_output_data_type_size);
    output_uav_desc.Buffer.FirstElement = output_buffer_regions[0].Offset() / m_output_data_type_size;
    output_uav_desc.Buffer.NumElements = static_cast<uint32_t>(output_buffer_regions[0].SizeInBytes() / m_output_data_type_size);

    auto output_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(
        m_heap->GetCPUDescriptorHandleForHeapStart(),
        2,
        d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    );

    d3d12Device->CreateUnorderedAccessView(output_buffer_regions[0].GetD3D12Resource(), nullptr, &output_uav_desc, output_cpu_handle);
}