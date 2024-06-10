#pragma once

#include <numeric>
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include "dml-operator.h"
#include "dml-execution-context.h"
#include "dml-managed-buffer.h"

using Microsoft::WRL::ComPtr;

class DmlGraphOperator : public DmlOperator
{
public:
    DmlGraphOperator(dml::Graph& scope, dml::Expression expression, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* command_recorder, Dml::ExecutionContext* executionContext, Dml::DmlGpuAllocator& allocator);
    void RecordDispatch(
        ID3D12GraphicsCommandList* command_list,
        const Dml::D3D12BufferRegion& temporary_buffer_region) final;

    void UpdateBindings(
        ID3D12Device* d3d12Device,
        void** raw_input_data,
        void* raw_output_data,
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions) final;

    const void* GetRawInputData(int index) const final {
        if (index >= m_raw_input_data.size()) {
            return nullptr;
        }

        return m_raw_input_data[index];
    }

    const void* GetRawOutputData() const final { return m_raw_output_data; }

    uint64_t GetTemporaryResourceSize() const final { return m_compiledOp->GetBindingProperties().TemporaryResourceSize; }

private:
    ComPtr<Dml::DmlManagedBuffer> m_managedPersistentBuffer;
    ComPtr<ID3D12Resource> m_persistentResource;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_compiledOp;
    DML_BUFFER_BINDING m_persistentResourceBinding;
    DML_BINDING_DESC m_persistentResourceBindingDesc {};
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_heap;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
    IDMLCommandRecorder* m_dmlCommandRecorder;
    Optional<Dml::DmlBuffer> m_persistentBuffer;
    DML_BINDING_TABLE_DESC m_binding_table_desc {};
    std::vector<void*> m_raw_input_data;
    void* m_raw_output_data = nullptr;
};