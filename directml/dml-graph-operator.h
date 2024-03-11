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
    DmlGraphOperator(dml::Graph& scope, dml::Expression expression, std::shared_ptr<Dml::ExecutionContext> executionContext, Dml::DmlGpuAllocator& allocator);
    void Execute(
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions) final;

private:
    std::shared_ptr<Dml::ExecutionContext> m_executionContext;
    ComPtr<Dml::DmlManagedBuffer> m_managedPersistentBuffer;
    ComPtr<ID3D12Resource> m_persistentResource;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_compiledOp;
    DML_BUFFER_BINDING m_persistentResourceBinding;
    DML_BINDING_DESC m_persistentResourceBindingDesc {};
};