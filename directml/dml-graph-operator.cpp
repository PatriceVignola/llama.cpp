#include <wil/wrl.h>
#include "dml-graph-operator.h"
#include "dml-allocator-rounding-mode.h"
#include "dml-managed-buffer.h"

DmlGraphOperator::DmlGraphOperator(
    dml::Graph& scope,
    dml::Expression expression,
    std::shared_ptr<Dml::ExecutionContext> executionContext,
    Dml::DmlGpuAllocator& allocator
) {
    m_executionContext = std::move(executionContext);
    m_compiledOp = scope.Compile(DML_EXECUTION_FLAG_NONE, {expression});

    uint64_t persistentResourceSize = m_compiledOp->GetBindingProperties().PersistentResourceSize;
    if (persistentResourceSize > 0)
    {
        auto buffer = allocator.AllocateDefaultBuffer(persistentResourceSize, Dml::AllocatorRoundingMode::Disabled);
        m_persistentResourceBinding = buffer.GetBufferBinding();
        m_managedPersistentBuffer = wil::MakeOrThrow<Dml::DmlManagedBuffer>(std::move(buffer));
        m_persistentResourceBindingDesc = DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &m_persistentResourceBinding };
    }

    DML_BINDING_DESC initInputBindings{};

    m_executionContext->InitializeOperator(
        m_compiledOp.Get(),
        m_persistentResourceBindingDesc,
        initInputBindings);
}

void DmlGraphOperator::Execute(
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions)
{
    // Queue references to objects which must be kept alive until resulting GPU work completes
    m_executionContext->QueueReference(m_compiledOp.Get());
    m_executionContext->QueueReference(m_managedPersistentBuffer.Get());

    auto FillBindingsFromBuffers = [](auto& bufferBindings, auto& bindingDescs, const std::vector<Dml::D3D12BufferRegion>& bufferRegions)
    {
        for (auto& bufferRegion : bufferRegions)
        {
            bufferBindings.push_back(bufferRegion.GetBufferBinding());
            bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
        }
    };

    // Bind the inputs
    std::vector<DML_BUFFER_BINDING> inputBufferBindings;
    inputBufferBindings.reserve(input_buffer_regions.size());
    std::vector<DML_BINDING_DESC> inputBindings;
    inputBindings.reserve(input_buffer_regions.size());
    FillBindingsFromBuffers(inputBufferBindings, inputBindings, input_buffer_regions);

    // Bind the outputs
    std::vector<DML_BUFFER_BINDING> outputBufferBindings;
    outputBufferBindings.reserve(output_buffer_regions.size());
    std::vector<DML_BINDING_DESC> outputBindings;
    outputBindings.reserve(output_buffer_regions.size());
    FillBindingsFromBuffers(outputBufferBindings, outputBindings, output_buffer_regions);

    // Execute the operator
    m_executionContext->ExecuteGraphOperator(m_compiledOp.Get(), m_persistentResourceBindingDesc, inputBindings, outputBindings);
}