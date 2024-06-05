#pragma once

#include <numeric>
#include <d3d12.h>
#include <wrl/client.h>
#include <wrl/implements.h>
#include <vector>
#include "dml-operator.h"
#include "dml-buffer-region.h"
#include "dml-execution-context.h"

using Microsoft::WRL::ComPtr;

class DmlQuantTensorPreprocessor
{
public:
    DmlQuantTensorPreprocessor(
        ID3D12Device* d3d12Device,
        Dml::ExecutionContext* executionContext,
        const dml::TensorDimensions& inputSizes,
        const dml::TensorStrides& inputStrides);

    void Execute(
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions);

private:
    struct Constants {
        uint32_t inputSizes[4];
        uint32_t inputStrides[4];
        uint32_t elementCount;
        uint32_t startIndex;
    };

    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    Constants m_constants;
    Dml::ExecutionContext* m_executionContext;
};