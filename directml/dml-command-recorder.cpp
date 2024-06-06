// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define NOMINMAX

#include <assert.h>
#include <wil/result.h>
#include <wil/wrl.h>
#include "dml-command-recorder.h"
#include "dml-command-queue.h"
#include "dml-gpu-allocator.h"
#include "dml-managed-buffer.h"

using namespace Dml;

DmlCommandRecorder::DmlCommandRecorder(
    ID3D12Device* d3dDevice,
    IDMLDevice* dmlDevice,
    std::shared_ptr<CommandQueue> commandQueue)
    : m_queue(std::move(commandQueue)),
      m_d3dDevice(d3dDevice),
      m_dmlDevice(dmlDevice),
      m_descriptorPool(d3dDevice, 2048)
{
    THROW_IF_FAILED(dmlDevice->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&m_initializer)));
    THROW_IF_FAILED(dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_recorder)));
}

DmlCommandRecorder::~DmlCommandRecorder()
{
    // Detach the threads to avoid crashes when terminating the program
    for (auto& resetThread : m_resetThreads)
    {
        if (resetThread)
        {
            resetThread->detach();
        }
    }
}

void DmlCommandRecorder::SetAllocator(std::shared_ptr<DmlGpuAllocator> allocator)
{
    m_allocator = std::move(allocator);
}

void DmlCommandRecorder::InitializeOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistentResourceBinding,
    const DML_BINDING_DESC& inputArrayBinding)
{
    // Reset the initializer to reference the input operator.
    IDMLCompiledOperator* ops[] = { op };
    THROW_IF_FAILED(m_initializer->Reset(ARRAYSIZE(ops), ops));

    DML_BINDING_PROPERTIES initBindingProps = m_initializer->GetBindingProperties();

    const uint32_t numDescriptors = initBindingProps.RequiredDescriptorCount;
    DescriptorRange descriptorRange = m_descriptorPool.AllocDescriptors(
        numDescriptors,
        m_queue->GetNextCompletionEvent());

    // Create a binding table for initialization.
    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_initializer.Get();
    bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
    bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
    bindingTableDesc.SizeInDescriptors = numDescriptors;

    ComPtr<IDMLBindingTable> bindingTable;
    THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

    // Create a temporary resource for initializing the op, if it's required.
    UINT64 temporaryResourceSize = initBindingProps.TemporaryResourceSize;
    if (temporaryResourceSize > 0)
    {
        if (m_temporaryBuffer && m_temporaryBuffer->SizeInBytes() < temporaryResourceSize) {
            // The temporary buffer is not big enough, so delete it and create a new one
            auto managedTemporaryBuffer = wil::MakeOrThrow<Dml::DmlManagedBuffer>(std::move(*m_temporaryBuffer));
            m_queue->QueueReference(managedTemporaryBuffer.Get(), true);
            m_temporaryBuffer = {};
        }

        if (!m_temporaryBuffer) {
            m_temporaryBuffer = m_allocator->AllocateDefaultBuffer(temporaryResourceSize);
        }

        // Bind the temporary resource.
        DML_BUFFER_BINDING bufferBinding = m_temporaryBuffer->GetBufferBinding();
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        bindingTable->BindTemporaryResource(&bindingDesc);
    }

    // Bind inputs, if provided.
    if (inputArrayBinding.Type != DML_BINDING_TYPE_NONE)
    {
        // An operator with inputs to bind MUST use a BUFFER_ARRAY.
        assert(inputArrayBinding.Type == DML_BINDING_TYPE_BUFFER_ARRAY);
        bindingTable->BindInputs(1, &inputArrayBinding);
    }

    // Bind the persistent resource, which is an output of initialization.
    if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE)
    {
        // Persistent resources MUST be bound as buffers.
        assert(persistentResourceBinding.Type == DML_BINDING_TYPE_BUFFER);
        bindingTable->BindOutputs(1, &persistentResourceBinding);
    }

    // Record the initialization work.
    SetDescriptorHeap(descriptorRange.heap);
    m_recorder->RecordDispatch(m_currentCommandList.Get(), m_initializer.Get(), bindingTable.Get());
    m_operationsRecordedInCurrentCommandList = true;

    // Barrier if there's an output (i.e. persistent resource), or if any temps are used.
    if ((persistentResourceBinding.Type != DML_BINDING_TYPE_NONE) ||
        (temporaryResourceSize > 0))
    {
        auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
        m_currentCommandList->ResourceBarrier(1, &uav);
    }
}

void DmlCommandRecorder::ExecuteGraphOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistentResourceBinding,
    const std::vector<DML_BINDING_DESC>& inputBindings,
    const std::vector<DML_BINDING_DESC>& outputBindings)
{
    DML_BINDING_PROPERTIES execBindingProps = op->GetBindingProperties();

    const uint32_t numDescriptors = execBindingProps.RequiredDescriptorCount;
    DescriptorRange descriptorRange = m_descriptorPool.AllocDescriptors(
        numDescriptors,
        m_queue->GetNextCompletionEvent());

    // Create a binding table for execution.
    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = op;
    bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
    bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
    bindingTableDesc.SizeInDescriptors = numDescriptors;

    ComPtr<IDMLBindingTable> bindingTable;
    THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

    // Create a temporary resource for executing the op, if it's required.
    UINT64 temporaryResourceSize = execBindingProps.TemporaryResourceSize;
    if (temporaryResourceSize > 0)
    {
        if (m_temporaryBuffer && m_temporaryBuffer->SizeInBytes() < temporaryResourceSize) {
            // The temporary buffer is not big enough, so delete it and create a new one
            auto managedTemporaryBuffer = wil::MakeOrThrow<Dml::DmlManagedBuffer>(std::move(*m_temporaryBuffer));
            m_queue->QueueReference(managedTemporaryBuffer.Get(), true);
            m_temporaryBuffer = {};
        }

        if (!m_temporaryBuffer) {
            m_temporaryBuffer = m_allocator->AllocateDefaultBuffer(temporaryResourceSize);
        }

        // Bind the temporary resource.
        DML_BUFFER_BINDING bufferBinding = m_temporaryBuffer->GetBufferBinding();
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        bindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE)
    {
        bindingTable->BindPersistentResource(&persistentResourceBinding);
    }

    bindingTable->BindInputs(static_cast<uint32_t>(inputBindings.size()), inputBindings.data());
    bindingTable->BindOutputs(static_cast<uint32_t>(outputBindings.size()), outputBindings.data());

    // Record the execution work.
    SetDescriptorHeap(descriptorRange.heap);
    m_recorder->RecordDispatch(m_currentCommandList.Get(), op, bindingTable.Get());
    m_operationsRecordedInCurrentCommandList = true;

    // Barrier all outputs.
    #pragma warning(push)
    #pragma warning(disable: 6387)
    auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    m_currentCommandList->ResourceBarrier(1, &uav);
    #pragma warning(pop)
}

inline uint32_t CeilDivide(uint32_t dividend, uint32_t divisor)
{
    uint64_t temp = static_cast<uint64_t>(dividend) + divisor - 1;
    return static_cast<uint32_t>(temp / divisor);
}

static void GetNextDispatchSize(
    uint32_t elementCount,
    uint32_t numThreads,
    _Out_ uint32_t& dispatch,
    _Out_ uint32_t& pendingElementCount
)
{
    // Max threads per workgroup is 2^10 (1024). Max dispatch per dimension is 2^16. Taken together, we can dispatch a maximum of
    // 2^26 (268,435,456) threads along a single dimension. This should suffice for a majority of the workload. Therefore, even
    // though it is possible to dispatch up to (2^16)^3 workgroups simultaneously, we stick to the simpler 1D dispatch alternative.
    assert(numThreads <= D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP);

    const uint32_t maxThreadsPerDispatch = numThreads * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

    // Compute max dispatchable elements
    const uint32_t availableThreadCount = std::min(elementCount, maxThreadsPerDispatch);

    // Compute required thread group count
    uint32_t workGroupCount1D = CeilDivide(availableThreadCount, numThreads);

    // Compute min dispatch size
    dispatch = workGroupCount1D;

    // With the dispatch size computed, compute the dispatched element count
    const uint32_t dispatchedElementCount = workGroupCount1D * numThreads;

    // Update the pending element count
    pendingElementCount = (dispatchedElementCount < elementCount) ? elementCount - dispatchedElementCount : 0;
}

void DmlCommandRecorder::ExecuteCustomOperator(
    ID3D12RootSignature* root_signature,
    ID3D12PipelineState* pipeline_state,
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions,
    const void* constants,
    uint32_t total_element_count,
    uint32_t constant_count)
{
    DescriptorRange descriptorRange = m_descriptorPool.AllocDescriptors(
        input_buffer_regions.size() + output_buffer_regions.size(),
        m_queue->GetNextCompletionEvent());

    // Set the root signature and pipeline state
    m_currentCommandList->SetComputeRootSignature(root_signature);
    m_currentCommandList->SetPipelineState(pipeline_state);

    uint32_t uav_view_index = 0;
    for (const auto& input_buffer_region : input_buffer_regions) {
        m_currentCommandList->SetComputeRootUnorderedAccessView(uav_view_index++, input_buffer_region.GetD3D12Resource()->GetGPUVirtualAddress() + input_buffer_region.Offset());
    }

    for (const auto& output_buffer_region : output_buffer_regions) {
        m_currentCommandList->SetComputeRootUnorderedAccessView(uav_view_index++, output_buffer_region.GetD3D12Resource()->GetGPUVirtualAddress() + output_buffer_region.Offset());
    }

    auto pendingElementCount = total_element_count;

    // Dispatch up to the maximum number of threads per iteration until
    // all elements are completed
    while (pendingElementCount > 0)
    {
        const uint32_t startIndex = total_element_count - pendingElementCount;

        uint32_t dispatchSizeX;

        GetNextDispatchSize(
            pendingElementCount,
            256,
            dispatchSizeX,
            pendingElementCount
        );

        // Set root constants
        m_currentCommandList->SetComputeRoot32BitConstants(
            input_buffer_regions.size() + output_buffer_regions.size(), // root parameter index
            constant_count, // Constant count
            constants,
            0 // offset
        );

        // Set the start index
        m_currentCommandList->SetComputeRoot32BitConstants(
            input_buffer_regions.size() + output_buffer_regions.size(), // root parameter index
            1, // Constant count
            &startIndex,
            constant_count - 1 // offset
        );

        m_currentCommandList->Dispatch(dispatchSizeX, 1, 1);
    }

    // Record the execution work.
    SetDescriptorHeap(descriptorRange.heap);
    m_operationsRecordedInCurrentCommandList = true;

    // Barrier all outputs.
    std::vector<D3D12_RESOURCE_BARRIER> output_barriers(output_buffer_regions.size());
    for (int i = 0; i < output_buffer_regions.size(); ++i) {
        output_barriers[i] = CD3DX12_RESOURCE_BARRIER::UAV(output_buffer_regions[i].GetD3D12Resource());
    }
    m_currentCommandList->ResourceBarrier(output_barriers.size(), output_barriers.data());
}

void DmlCommandRecorder::ExecuteCustomOperatorByGroup(
    ID3D12RootSignature* root_signature,
    ID3D12PipelineState* pipeline_state,
    const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
    const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions,
    const void* constants,
    uint32_t constant_count,
    uint32_t groupCountX,
    uint32_t groupCountY,
    uint32_t groupCountZ)
{
    DescriptorRange descriptorRange = m_descriptorPool.AllocDescriptors(
        input_buffer_regions.size() + output_buffer_regions.size(),
        m_queue->GetNextCompletionEvent());

    // Set the root signature and pipeline state
    m_currentCommandList->SetComputeRootSignature(root_signature);
    m_currentCommandList->SetPipelineState(pipeline_state);

    uint32_t uav_view_index = 0;
    for (const auto& input_buffer_region : input_buffer_regions) {
        m_currentCommandList->SetComputeRootUnorderedAccessView(uav_view_index++, input_buffer_region.GetD3D12Resource()->GetGPUVirtualAddress() + input_buffer_region.Offset());
    }

    for (const auto& output_buffer_region : output_buffer_regions) {
        m_currentCommandList->SetComputeRootUnorderedAccessView(uav_view_index++, output_buffer_region.GetD3D12Resource()->GetGPUVirtualAddress() + output_buffer_region.Offset());
    }

    // Set root constants
    m_currentCommandList->SetComputeRoot32BitConstants(
        input_buffer_regions.size() + output_buffer_regions.size(), // root parameter index
        constant_count, // Constant count
        constants,
        0 // offset
    );

    m_currentCommandList->Dispatch(groupCountX, groupCountY, groupCountZ);

    // Record the execution work.
    SetDescriptorHeap(descriptorRange.heap);
    m_operationsRecordedInCurrentCommandList = true;

    // Barrier all outputs.
    std::vector<D3D12_RESOURCE_BARRIER> output_barriers(output_buffer_regions.size());
    for (int i = 0; i < output_buffer_regions.size(); ++i) {
        output_barriers[i] = CD3DX12_RESOURCE_BARRIER::UAV(output_buffer_regions[i].GetD3D12Resource());
    }
    m_currentCommandList->ResourceBarrier(output_barriers.size(), output_barriers.data());
}

void DmlCommandRecorder::CopyBufferRegion(
    ID3D12Resource* dstBuffer,
    uint64_t dstOffset,
    ID3D12Resource* srcBuffer,
    uint64_t srcOffset,
    uint64_t byteCount)
{
    m_currentCommandList->CopyBufferRegion(dstBuffer, dstOffset, srcBuffer, srcOffset, byteCount);
    m_operationsRecordedInCurrentCommandList = true;
}

void DmlCommandRecorder::FillBufferWithPattern(ID3D12Resource* dstBuffer, uint64_t offset, const uint8_t* data, uint64_t size)
{
    // The fill pattern for ClearUnorderedAccessViewUint is 16 bytes.
    union
    {
        uint32_t integers[4];
        uint8_t bytes[16];
    } fillPattern = {};

    assert(ARRAYSIZE(fillPattern.bytes) == 16);
    assert(size <= ARRAYSIZE(fillPattern.bytes)); // No element is expected larger than 128 bits (e.g. complex128).

    if (size != 0)
    {
        assert(ARRAYSIZE(fillPattern.bytes) % size == 0); // Should fit evenly into 16 bytes (e.g. uint8, float16, uint32, float64...).

        // Repeat the value multiple times into the pattern buffer.
        size_t valueIndex = 0;
        for (uint8_t& p : fillPattern.bytes)
        {
            p = data[valueIndex++];
            valueIndex = (valueIndex == size) ? 0 : valueIndex;
        }
    }
    // Else just leave fill pattern as zeroes.

    // Create a RAW buffer UAV over the resource.
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    uavDesc.Buffer.FirstElement = static_cast<uint32_t>(offset / sizeof(uint32_t));
    uavDesc.Buffer.NumElements = static_cast<uint32_t>(dstBuffer->GetDesc().Width / sizeof(uint32_t));
    uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

    const uint32_t neededDescriptorCount = 1;
    DescriptorRange descriptorRangeCpu = m_descriptorPool.AllocDescriptors(neededDescriptorCount, m_queue->GetNextCompletionEvent(), D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    DescriptorRange descriptorRangeGpu = m_descriptorPool.AllocDescriptors(neededDescriptorCount, m_queue->GetNextCompletionEvent(), D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    m_d3dDevice->CreateUnorderedAccessView(dstBuffer, nullptr, &uavDesc, descriptorRangeCpu.cpuHandle);
    m_d3dDevice->CreateUnorderedAccessView(dstBuffer, nullptr, &uavDesc, descriptorRangeGpu.cpuHandle);

    SetDescriptorHeap(descriptorRangeGpu.heap);

    // Record a ClearUAV onto the command list.
    m_currentCommandList->ClearUnorderedAccessViewUint(
        descriptorRangeGpu.gpuHandle,
        descriptorRangeCpu.cpuHandle,
        dstBuffer,
        fillPattern.integers,
        0,
        nullptr);
    m_operationsRecordedInCurrentCommandList = true;

    // Barrier all outputs.
    #pragma warning(push)
    #pragma warning(disable: 6387)
    auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    m_currentCommandList->ResourceBarrier(1, &uav);
    #pragma warning(pop)
}

void DmlCommandRecorder::ExecuteCommandList(
    ID3D12GraphicsCommandList* commandList,
    _Outptr_ ID3D12Fence** fence,
    _Out_ uint64_t* completionValue
    )
{
    if (!m_operationsRecordedInCurrentCommandList)
    {
        // The caller can re-use relevant resources after the next set of work to be
        // flushed has completed.  Its command list hasn't been executed yet, just batched.
        GpuEvent gpuEvent = m_queue->GetNextCompletionEvent();
        gpuEvent.fence.CopyTo(fence);
        *completionValue = gpuEvent.fenceValue;

        m_queue->ExecuteCommandList(commandList);

        // The fence value at which the current command allocator may be re-used will now be higher
        m_allocatorRing.back().completionEvent = m_queue->GetNextCompletionEvent();

        // Fail early if something horrifying happens
        THROW_IF_FAILED(m_dmlDevice->GetDeviceRemovedReason());
        THROW_IF_FAILED(m_d3dDevice->GetDeviceRemovedReason());

        return;
    }

    // Remember the descriptor heap and apply it to the next command list.  This avoids unnecessarily setting it onto
    // the D3D object lazily at a point when the operation may not be parallelized with GPU work.
    auto heap = m_currentDescriptorHeap;

    // Execute work in the current command list plus provided command list while closing the recorder.
    CloseAndExecute(commandList);
    Open();

    // Reset the descriptor heap opportunistically per above comment
    SetDescriptorHeap(heap);

    GpuEvent gpuEvent = m_queue->GetCurrentCompletionEvent();
    gpuEvent.fence.CopyTo(fence);
    *completionValue = gpuEvent.fenceValue;
}

ComPtr<ID3D12GraphicsCommandList> DmlCommandRecorder::GetCommandList()
{
    // Assume operations are added by the caller after this returns
    m_operationsRecordedInCurrentCommandList = true;
    return m_currentCommandList;
}

void DmlCommandRecorder::ResourceBarrier(const std::vector<D3D12_RESOURCE_BARRIER>& barriers)
{
    m_currentCommandList->ResourceBarrier(static_cast<uint32_t>(barriers.size()), barriers.data());
    m_operationsRecordedInCurrentCommandList = true;
}

void DmlCommandRecorder::AddUAVBarrier()
{
    #pragma warning(suppress: 6387)
    auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    m_currentCommandList->ResourceBarrier(1, &barrier);
    m_operationsRecordedInCurrentCommandList = true;
}

void DmlCommandRecorder::Open()
{
    assert(m_currentDescriptorHeap == nullptr);

    if (m_currentCommandList)
    {
        if (m_resetThreads.front())
        {
            m_resetThreads.front()->join();
        }

        // Rotate the reset threads to the left
        for (uint32_t i = 0; i < m_resetThreads.size() - 1; ++i) {
            m_resetThreads[i] = std::move(m_resetThreads[i + 1]);
        }

        // Rotate the allocators to the left
        auto firstAllocator = std::move(m_allocatorRing.front());
        for (uint32_t i = 0; i < m_allocatorRing.size() - 1; ++i)
        {
            m_allocatorRing[i] = std::move(m_allocatorRing[i + 1]);
        }
        m_allocatorRing.back() = std::move(firstAllocator);

        // Rotate the command lists to the left
        auto firstCommandList = std::move(m_commandListRing.front());
        for (uint32_t i = 0; i < m_commandListRing.size() - 1; ++i)
        {
            m_commandListRing[i] = std::move(m_commandListRing[i + 1]);
        }
        m_commandListRing.back() = std::move(firstCommandList);

        // The newest dirty allocator is now located before the last element in the ring buffer, so start resetting it
        m_resetThreads.back() = std::make_unique<std::thread>([cachedAllocator = m_allocatorRing[m_allocatorRing.size() - 2], cachedCommandList = m_commandListRing[m_commandListRing.size() - 2]]() {
            cachedAllocator.completionEvent.WaitForSignal();
            THROW_IF_FAILED(cachedAllocator.allocator->Reset());
            THROW_IF_FAILED(cachedCommandList->Reset(cachedAllocator.allocator.Get(), nullptr));
        });
    }
    else
    {
        assert(m_commandListRing.size() == m_allocatorRing.size());

        for (uint32_t i = 0; i < m_commandListRing.size(); ++i)
        {
            THROW_IF_FAILED(m_d3dDevice->CreateCommandAllocator(
                m_queue->GetType(),
                IID_PPV_ARGS(m_allocatorRing[i].allocator.ReleaseAndGetAddressOf())));

            THROW_IF_FAILED(m_d3dDevice->CreateCommandList(
                0,
                m_queue->GetType(),
                m_allocatorRing[i].allocator.Get(),
                nullptr,
                IID_PPV_ARGS(m_commandListRing[i].ReleaseAndGetAddressOf())));
        }
    }

    m_currentCommandList = m_commandListRing.back();
    m_allocatorRing.back().completionEvent = m_queue->GetNextCompletionEvent();
}

void DmlCommandRecorder::CloseAndExecute()
{
    CloseAndExecute(nullptr);
}

void DmlCommandRecorder::CloseAndExecute(_In_opt_ ID3D12GraphicsCommandList* commandList)
{
    THROW_IF_FAILED(m_currentCommandList->Close());

    ID3D12CommandList* commandListsToExecute[2] = {};
    uint32_t commandListsToExecuteCount = 0;

    if (m_operationsRecordedInCurrentCommandList)
    {
        commandListsToExecute[commandListsToExecuteCount++] = m_currentCommandList.Get();
    }

    if (commandList)
    {
        commandListsToExecute[commandListsToExecuteCount++] = commandList;
    }

    if (commandListsToExecuteCount > 0)
    {
        m_queue->ExecuteCommandLists(commandListsToExecute, commandListsToExecuteCount);
    }

    m_operationsRecordedInCurrentCommandList = false;

    // The descriptor heap must be set on the command list the next time it's opened.
    m_currentDescriptorHeap = nullptr;

    // Fail early if something horrifying happens
    THROW_IF_FAILED(m_dmlDevice->GetDeviceRemovedReason());
    THROW_IF_FAILED(m_d3dDevice->GetDeviceRemovedReason());
}

void DmlCommandRecorder::SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap)
{
    if (descriptorHeap != nullptr && descriptorHeap != m_currentDescriptorHeap)
    {
        m_currentDescriptorHeap = descriptorHeap;

        ID3D12DescriptorHeap* descriptorHeaps[] = { descriptorHeap };
        m_currentCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);
    }
}
