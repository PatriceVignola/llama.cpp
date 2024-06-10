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
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions,
        const Dml::D3D12BufferRegion& temporary_buffer_region) final;

    uint64_t GetTemporaryResourceSize() final { return m_compiledOp->GetBindingProperties().TemporaryResourceSize; }

private:
    void ExecuteGraphOperator(
        ID3D12GraphicsCommandList* command_list,
        const std::vector<DML_BINDING_DESC>& inputBindings,
        const std::vector<DML_BINDING_DESC>& outputBindings,
        const Dml::D3D12BufferRegion& temporary_buffer_region);

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
};