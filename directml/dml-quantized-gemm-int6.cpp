#include <d3dx12.h>
#include <assert.h>
#include <wil/result.h>
#include "dml-quantized-gemm-int6.h"

// The value in dml-quantized-gemm-int6.hlsl needs to also be changed if this is changed here
#define K_QUANTS_PER_ITERATION 2

namespace DmlQuantizedGemmInt6_16_32
{
    #include "generated-shaders/dml-quantized-gemm-int6-16-32.h"
}

namespace DmlQuantizedGemmInt6_32_16
{
    #include "generated-shaders/dml-quantized-gemm-int6-32-16.h"
}

namespace DmlQuantizedGemmInt6_16_16
{
    #include "generated-shaders/dml-quantized-gemm-int6-16-16.h"
}

namespace DmlQuantizedGemmInt6_32_32
{
    #include "generated-shaders/dml-quantized-gemm-int6-32-32.h"
}

DmlQuantizedGemmInt6Operator::DmlQuantizedGemmInt6Operator(
    ID3D12Device* d3d12Device,
    Dml::ExecutionContext* executionContext,
    uint32_t nrows,
    uint32_t ncols,
    DML_TENSOR_DATA_TYPE input_data_type,
    DML_TENSOR_DATA_TYPE output_data_type)
        : m_device(d3d12Device)
        , m_executionContext(executionContext)
{
    m_constants.nrows = nrows;
    m_constants.ncols = ncols;

    const int ny = 2 / K_QUANTS_PER_ITERATION;
    m_groupCount = (nrows + ny - 1) / ny;

    // Compute root signature.
    const int uavCount = 3;
    std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
    rootParameters.resize(uavCount + 1);

    for (UINT i = 0; i < uavCount; i++)
    {
        rootParameters[i].InitAsUnorderedAccessView(i);
    }

    const int constantCount = 2;
    rootParameters[uavCount].InitAsConstants(constantCount, 0);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
    desc.Init_1_1(static_cast<uint32_t>(rootParameters.size()), rootParameters.data());

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

    if (input_data_type == DML_TENSOR_DATA_TYPE_FLOAT16 && output_data_type == DML_TENSOR_DATA_TYPE_FLOAT32) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlQuantizedGemmInt6_16_32::g_main, sizeof(DmlQuantizedGemmInt6_16_32::g_main));
    } else if (input_data_type == DML_TENSOR_DATA_TYPE_FLOAT32 && output_data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlQuantizedGemmInt6_32_16::g_main, sizeof(DmlQuantizedGemmInt6_32_16::g_main));
    } else if (input_data_type == DML_TENSOR_DATA_TYPE_FLOAT16 && output_data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlQuantizedGemmInt6_16_16::g_main, sizeof(DmlQuantizedGemmInt6_16_16::g_main));
    } else if (input_data_type == DML_TENSOR_DATA_TYPE_FLOAT32 && output_data_type == DML_TENSOR_DATA_TYPE_FLOAT32) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlQuantizedGemmInt6_32_32::g_main, sizeof(DmlQuantizedGemmInt6_32_32::g_main));
    } else {
        THROW_HR(E_NOTIMPL);
    }

    THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));
}

void DmlQuantizedGemmInt6Operator::Execute(
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions)
{
    // Execute the operator
    m_executionContext->ExecuteCustomOperatorByGroup(
        m_rootSignature.Get(),
        m_pipelineState.Get(),
        input_buffer_regions,
        output_buffer_regions,
        &m_constants,
        sizeof(m_constants) / sizeof(uint32_t),
        m_groupCount,
        1,
        1);
}