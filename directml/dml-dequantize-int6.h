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
        const Dml::D3D12BufferRegion& temporary_buffer_region) final;

    uint64_t GetTemporaryResourceSize() const final { return 0; }

    void UpdateBindings(
        ID3D12Device* d3d12Device,
        void** raw_input_data,
        void* raw_output_data,
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions) final;

    const void* GetRawInputData(int index) const final {
        if (index >= m_raw_input_data.size()) {
            THROW_HR(E_UNEXPECTED);
        }

        return m_raw_input_data[index];
    }

    const void* GetRawOutputData() const final { return m_raw_output_data; }

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
    std::array<void*, 2> m_raw_input_data{};
    void* m_raw_output_data = nullptr;
    size_t m_output_data_type_size;
};