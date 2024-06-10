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
    std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
    rootParameters.resize(uavCount + 1);

    for (UINT i = 0; i < uavCount; i++)
    {
        rootParameters[i].InitAsUnorderedAccessView(i);
    }

    // TODO (pavignol): Clean me up
    const int constantCount = 0;
    rootParameters[uavCount].InitAsConstants(constantCount, 0);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
    desc.Init_1_1(static_cast<uint32_t>(rootParameters.size()), rootParameters.data());

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
    } else if (output_data_type == DML_TENSOR_DATA_TYPE_FLOAT32) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlDequantizeInt6_32::g_main, sizeof(DmlDequantizeInt6_32::g_main));
    } else {
        THROW_HR(E_NOTIMPL);
    }

    THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));
}

void DmlDequantizeInt6Operator::RecordDispatch(
    ID3D12GraphicsCommandList* command_list,
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions,
    const Dml::D3D12BufferRegion& temporary_buffer_region)
{
    // Execute the operator
    Dml::DmlCommandRecorder::RecordCustomOperatorDispatchByGroup(
        command_list,
        m_rootSignature.Get(),
        m_pipelineState.Get(),
        m_heap.Get(),
        input_buffer_regions,
        output_buffer_regions,
        &m_constants,
        sizeof(m_constants) / sizeof(uint32_t),
        m_groupCount,
        1,
        1);
}