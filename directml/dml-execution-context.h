// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "dml-gpu-event.h"
#include "dml-command-recorder.h"
#include "dml-gpu-allocator.h"

namespace Dml
{
    class CommandQueue;

    // Asynchronously performs GPU work, and automatically manages command list recording and submission to queues.
    // Work submitted to the ExecutionContext is typically recorded onto a command list and may not immediately begin
    // execution on the GPU. Call Flush() to force all recorded work to be submitted to the command queue for execution
    // on the GPU.
    class ExecutionContext
    {
    public:
        // Constructs an ExecutionContext that executes on the supplied queue.
        ExecutionContext(
            ID3D12Device* d3d12Device,
            IDMLDevice* dmlDevice,
            ID3D12CommandQueue* queue);

        void SetAllocator(std::weak_ptr<DmlGpuAllocator> allocator);

        // Waits for flushed work, discards unflushed work, and discards associated references to
        // prevent circular references.  Must be the last call on the object before destruction.
        void Close();

        // Queues a CopyBufferRegion (see ID3D12GraphicsCommandList::CopyBufferRegion) for execution. Transition
        // barriers are automatically inserted to transition the source and destination resources to COPY_SOURCE and
        // COPY_DEST if necessary.
        void CopyBufferRegion(
            ID3D12Resource* dstBuffer,
            uint64_t dstOffset,
            D3D12_RESOURCE_STATES dstState,
            ID3D12Resource* srcBuffer,
            uint64_t srcOffset,
            D3D12_RESOURCE_STATES srcState,
            uint64_t byteCount);

        void FillBufferWithPattern(ID3D12Resource* dstBuffer, uint64_t offset, const uint8_t* data, uint64_t size);

        void InitializeOperator(
            IDMLCompiledOperator* op,
            const DML_BINDING_DESC& persistentResourceBinding,
            const DML_BINDING_DESC& inputArrayBinding);

        void ExecuteGraphOperator(
            IDMLCompiledOperator* op,
            const DML_BINDING_DESC& persistentResourceBinding,
            const std::vector<DML_BINDING_DESC>& inputBindings,
            const std::vector<DML_BINDING_DESC>& outputBindings);

        void ExecuteCustomOperator(
            ID3D12RootSignature* root_signature,
            ID3D12PipelineState* pipeline_state,
            const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
            const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions,
            const void* constants,
            uint32_t total_element_count,
            uint32_t constant_count);

        void ExecuteCommandList(
            ID3D12GraphicsCommandList* commandList,
            _Outptr_ ID3D12Fence** fence,
            _Out_ uint64_t* completionValue
            );

        void AddUAVBarrier();
        void ResourceBarrier(const std::vector<D3D12_RESOURCE_BARRIER>& barriers);

        void GetCommandListForRecordingAndInvalidateState(ID3D12GraphicsCommandList** commandList);

        // Forces all queued work to begin executing on the GPU. This method returns immediately and does not wait
        // for the submitted work to complete execution on the GPU.
        void Flush();

        // Returns an event which will become signaled when everything submitted to the execution context thus far has
        // completed execution on the GPU, including work that has yet to be flushed to the queue.
        GpuEvent GetCurrentCompletionEvent();

        // Adds a reference which will be released when queued GPU work is completed
        void QueueReference(IUnknown* object);

        // Release any accumulated references who corresponding GPU fence values have
        // been reached.
        void ReleaseCompletedReferences();

        D3D12_COMMAND_LIST_TYPE GetCommandListTypeForQueue() const;

        bool Closed() const { return m_closed; }

    private:
        ComPtr<ID3D12Device> m_d3dDevice;

        void SetCommandRecorder(DmlCommandRecorder* newRecorder);

        std::shared_ptr<CommandQueue> m_queue;

        DmlCommandRecorder* m_currentRecorder = nullptr;

        // Up to one of these is active at a time
        DmlCommandRecorder m_dmlRecorder;

        bool m_closed = false;
    };

} // namespace Dml
