#pragma once

#include <numeric>
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include "dml-operator.h"
#include "dml-buffer-region.h"
#include "dml-execution-context.h"

using Microsoft::WRL::ComPtr;

class DmlQuantizedGemmInt6Operator : public DmlOperator
{
public:
    DmlQuantizedGemmInt6Operator(
        ID3D12Device* d3d12Device,
        Dml::ExecutionContext* executionContext,
        uint32_t nrows,
        uint32_t ncols,
        DML_TENSOR_DATA_TYPE input_data_type,
        DML_TENSOR_DATA_TYPE output_data_type);

    void RecordDispatch(
        ID3D12GraphicsCommandList* command_list,
        const Dml::D3D12BufferRegion& temporary_buffer_region) final;

    void UpdateBindings(
        ID3D12Device* d3d12Device,
        void** raw_input_data,
        void** raw_output_data,
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions
    ) final {
        THROW_HR(E_NOTIMPL);
    }

    const void* GetRawInputData(int index) const final {
        THROW_HR(E_NOTIMPL);
    }

    const void* GetRawOutputData(int index) const final {
        THROW_HR(E_NOTIMPL);
    }

    uint64_t GetTemporaryResourceSize() const final { return 0; }
    bool LateBindingAllowed() const final { return true; }

private:
    struct Constants {
        uint32_t nrows;
        uint32_t ncols;
    };

    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    ComPtr<ID3D12DescriptorHeap> m_heap;
    Constants m_constants;
    uint32_t m_groupCount;
    Dml::ExecutionContext* m_executionContext;
};