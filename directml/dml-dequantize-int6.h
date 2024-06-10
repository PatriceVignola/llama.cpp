#pragma once

#include <numeric>
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include "dml-operator.h"
#include "dml-buffer-region.h"
#include "dml-execution-context.h"

using Microsoft::WRL::ComPtr;

class DmlDequantizeInt6Operator : public DmlOperator
{
public:
    DmlDequantizeInt6Operator(
        ID3D12Device* d3d12Device,
        Dml::ExecutionContext* executionContext,
        uint32_t k,
        DML_TENSOR_DATA_TYPE output_data_type);

    void RecordDispatch(
        ID3D12GraphicsCommandList* command_list,
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions,
        const Dml::D3D12BufferRegion& temporary_buffer_region) final;

    uint64_t GetTemporaryResourceSize() final { return 0; }

private:
    // TODO (pavignol): Clean me up
    struct Constants {
        uint32_t abc;
    };

    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    ComPtr<ID3D12DescriptorHeap> m_heap;
    Constants m_constants;
    uint32_t m_groupCount;
    Dml::ExecutionContext* m_executionContext;
};