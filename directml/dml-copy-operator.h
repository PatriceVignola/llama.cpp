#pragma once

#include <numeric>
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include "dml-operator.h"
#include "dml-buffer-region.h"
#include "dml-execution-context.h"

using Microsoft::WRL::ComPtr;

class DmlCopyOperator : public DmlOperator
{
public:
    DmlCopyOperator(
        ID3D12Device* d3d12Device,
        Dml::ExecutionContext* executionContext,
        const dml::TensorDesc& input_tensor_desc,
        const dml::TensorDesc& output_tensor_desc);

    void RecordDispatch(
        ID3D12GraphicsCommandList* command_list,
        const Dml::D3D12BufferRegion& temporary_buffer_region) final;

    uint64_t GetTemporaryResourceSize() const final { return 0; }

    void UpdateBindings(
        ID3D12Device* d3d12Device,
        void** raw_input_data,
        void** raw_output_data,
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions) final;

    const void* GetRawInputData(int index) const final {
        if (index > 0) {
            THROW_HR(E_UNEXPECTED);
        }

        return m_raw_input_data;
    }

    const void* GetRawOutputData(int index) const final {
        if (index > 0) {
            THROW_HR(E_UNEXPECTED);
        }

        return m_raw_output_data;
    }

    bool LateBindingAllowed() const final { return true; }

private:
    struct Constants {
        uint32_t inputSizes[4];
        uint32_t outputSizes[4];
        uint32_t inputStrides[4];
        uint32_t outputStrides[4];
        uint32_t elementCount;
        uint32_t startIndex;
    };

    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    ComPtr<ID3D12DescriptorHeap> m_heap;
    Constants m_constants;
    Dml::ExecutionContext* m_executionContext;
    size_t m_input_dtype_size;
    size_t m_output_dtype_size;
    void* m_raw_input_data = nullptr;
    void* m_raw_output_data = nullptr;
};