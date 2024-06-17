#include <wil/wrl.h>
#include "dml-graph-operator.h"
#include "dml-allocator-rounding-mode.h"
#include "dml-managed-buffer.h"

DmlGraphOperator::DmlGraphOperator(
    dml::Graph& scope,
    dml::Span<const dml::Expression> expressions,
    ID3D12Device* d3d12_device,
    IDMLDevice* dml_device,
    IDMLCommandRecorder* command_recorder,
    Dml::ExecutionContext* executionContext,
    Dml::DmlGpuAllocator& allocator
) {
    // TODO (pavignol): Add fp16 compute flag
    m_compiledOp = scope.Compile(DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE, expressions);
    m_dmlCommandRecorder = command_recorder;

    DML_BINDING_PROPERTIES binding_props = m_compiledOp->GetBindingProperties();

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    desc.NumDescriptors = binding_props.RequiredDescriptorCount;
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    THROW_IF_FAILED(d3d12_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(m_heap.ReleaseAndGetAddressOf())));

    // Create a binding table for execution.
    m_binding_table_desc.Dispatchable = m_compiledOp.Get();
    m_binding_table_desc.CPUDescriptorHandle = m_heap->GetCPUDescriptorHandleForHeapStart();
    m_binding_table_desc.GPUDescriptorHandle = m_heap->GetGPUDescriptorHandleForHeapStart();
    m_binding_table_desc.SizeInDescriptors = binding_props.RequiredDescriptorCount;
    THROW_IF_FAILED(dml_device->CreateBindingTable(&m_binding_table_desc, IID_PPV_ARGS(&m_bindingTable)));

    uint64_t persistentResourceSize = m_compiledOp->GetBindingProperties().PersistentResourceSize;
    if (persistentResourceSize > 0)
    {
        m_persistentBuffer = allocator.AllocateDefaultBuffer(persistentResourceSize, Dml::AllocatorRoundingMode::Disabled);
        m_persistentResourceBinding = m_persistentBuffer->GetBufferBinding();
        m_persistentResourceBindingDesc = DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &m_persistentResourceBinding };
    }

    DML_BINDING_DESC initInputBindings{};

    executionContext->InitializeOperator(
        m_compiledOp.Get(),
        m_persistentResourceBindingDesc,
        initInputBindings);
}

void DmlGraphOperator::RecordDispatch(
    ID3D12GraphicsCommandList* command_list,
    const Dml::D3D12BufferRegion& temporary_buffer_region)
{
    m_bindingTable->Reset(&m_binding_table_desc);

    if (m_persistentResourceBindingDesc.Type != DML_BINDING_TYPE_NONE)
    {
        m_bindingTable->BindPersistentResource(&m_persistentResourceBindingDesc);
    }

    // Use the temporary resource for executing the op, if it's required.
    uint64_t temporaryResourceSize = m_compiledOp->GetBindingProperties().TemporaryResourceSize;
    if (temporaryResourceSize > 0)
    {
        auto temporaryResourceBinding = temporary_buffer_region.GetBufferBinding();
        auto temporaryResourceBindingDesc = DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &temporaryResourceBinding };
        m_bindingTable->BindTemporaryResource(&temporaryResourceBindingDesc);
    }

    // TODO (pavignol): Try having a single descriptor heap with a descriptor range that spans all the operators in the command list
    // Set the descriptor heap
    ID3D12DescriptorHeap* descriptorHeaps[] = { m_heap.Get() };
    command_list->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);

    // Record the execution work.
    m_dmlCommandRecorder->RecordDispatch(command_list, m_compiledOp.Get(), m_bindingTable.Get());

    // Barrier all outputs.
    #pragma warning(push)
    #pragma warning(disable: 6387)
    auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    // TODO (pavignol): Use a dependency graph to figure out where to put the barriers
    command_list->ResourceBarrier(1, &uav);
    #pragma warning(pop)
}

void DmlGraphOperator::UpdateBindings(
    ID3D12Device* d3d12Device,
    void** raw_input_data,
    void** raw_output_data,
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions)
{
    m_raw_input_data.resize(input_buffer_regions.size());

    for (int i = 0; i < input_buffer_regions.size(); ++i) {
        m_raw_input_data[i] = raw_input_data[i];
    }

    m_raw_output_data.resize(output_buffer_regions.size());

    for (int i = 0; i < output_buffer_regions.size(); ++i) {
        m_raw_output_data[i] = raw_output_data[i];
    }

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

    // Bind the inputs and outputs (the temporary and persistent bindings shouldn't change)
    m_bindingTable->Reset(&m_binding_table_desc);
    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindings.size()), inputBindings.data());
    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindings.size()), outputBindings.data());
}