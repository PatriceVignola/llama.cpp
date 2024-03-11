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
    std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
    rootParameters.resize(uavCount + 1);

    for (UINT i = 0; i < uavCount; i++)
    {
        rootParameters[i].InitAsUnorderedAccessView(i);
    }

    const int constantCount = 18;
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

    auto input_dtype_size = get_data_type_size(input_tensor_desc.dataType);
    auto output_dtype_size = get_data_type_size(output_tensor_desc.dataType);

    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
    computePsoDesc.pRootSignature = m_rootSignature.Get();

    if (input_dtype_size == 2 && output_dtype_size == 4) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlCopy_16_32::g_main, sizeof(DmlCopy_16_32::g_main));
    } else if (input_dtype_size == 4 && output_dtype_size == 2) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlCopy_32_16::g_main, sizeof(DmlCopy_32_16::g_main));
    } else if (input_dtype_size == 2 && output_dtype_size == 2) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlCopy_16_16::g_main, sizeof(DmlCopy_16_16::g_main));
    } else if (input_dtype_size == 4 && output_dtype_size == 4) {
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(DmlCopy_32_32::g_main, sizeof(DmlCopy_32_32::g_main));
    } else {
        THROW_HR(E_NOTIMPL);
    }

    THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));
}

void DmlCopyOperator::Execute(
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