#include <d3dx12.h>
#include <assert.h>
#include <wil/result.h>
#include "dml-quant-tensor-preprocessor.h"

namespace DmlQuantTensorPreprocessorShader
{
    #include "generated-shaders/dml-quant-tensor-preprocessor.h"
}

DmlQuantTensorPreprocessor::DmlQuantTensorPreprocessor(
    ID3D12Device* d3d12Device,
    Dml::ExecutionContext* executionContext,
    const dml::TensorDimensions& inputSizes,
    const dml::TensorStrides& inputStrides)
        : m_device(d3d12Device)
        , m_executionContext(executionContext)
{
    std::copy_n(inputSizes.begin(), inputSizes.size(), m_constants.inputSizes);
    std::copy_n(inputStrides.begin(), inputStrides.size(), m_constants.inputStrides);
    m_constants.elementCount = std::accumulate(inputSizes.begin(), inputSizes.end(), 1u, std::multiplies<uint32_t>());

    // Compute root signature.
    const int uavCount = 3;
    std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
    rootParameters.resize(uavCount + 1);

    for (UINT i = 0; i < uavCount; i++)
    {
        rootParameters[i].InitAsUnorderedAccessView(i);
    }

    rootParameters[uavCount].InitAsConstants(sizeof(m_constants) / sizeof(uint32_t), 0);

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
    computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlQuantTensorPreprocessorShader::g_main, sizeof(DmlQuantTensorPreprocessorShader::g_main));

    THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));
}

void DmlQuantTensorPreprocessor::Execute(
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions)
{
    // Execute the operator
    m_executionContext->ExecuteCustomOperator(
        m_rootSignature.Get(),
        m_pipelineState.Get(),
        input_buffer_regions,
        output_buffer_regions,
        &m_constants,
        m_constants.elementCount,
        sizeof(m_constants) / sizeof(uint32_t)
    );
}