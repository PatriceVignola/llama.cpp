// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>
#include <array>
#include <thread>
#include <wrl/client.h>
#include <d3d12.h>
#include "include/DirectML.h"

#include "dml-gpu-event.h"
#include "dml-descriptor-pool.h"
#include "dml-command-queue.h"
#include "dml-gpu-allocator.h"
#include "dml_optional_extensions.h"

using Microsoft::WRL::ComPtr;

namespace Dml
{
    class DmlCommandRecorder
    {
    public:
        DmlCommandRecorder(
            ID3D12Device* d3dDevice,
            IDMLDevice* device,
            std::shared_ptr<CommandQueue> commandQueue);

        ~DmlCommandRecorder();

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

        void ExecuteCustomOperatorByGroup(
            ID3D12RootSignature* root_signature,
            ID3D12PipelineState* pipeline_state,
            const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
            const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions,
            const void* constants,
            uint32_t constant_count,
            uint32_t groupCountX,
            uint32_t groupCountY,
            uint32_t groupCountZ);

        void CopyBufferRegion(
            ID3D12Resource* dstBuffer,
            uint64_t dstOffset,
            ID3D12Resource* srcBuffer,
            uint64_t srcOffset,
            uint64_t byteCount);

        void FillBufferWithPattern(ID3D12Resource* dstBuffer, uint64_t offset, const uint8_t* data, uint64_t size);

        void ExecuteCommandList(
            ID3D12GraphicsCommandList* commandList,
            _Outptr_ ID3D12Fence** fence,
            _Out_ uint64_t* completionValue);

        ComPtr<ID3D12GraphicsCommandList> GetCommandList();

        void ResourceBarrier(const std::vector<D3D12_RESOURCE_BARRIER>& barriers);
        void AddUAVBarrier();

        void Open();
        void CloseAndExecute();

        void SetAllocator(std::shared_ptr<DmlGpuAllocator> allocator);

        bool HasUnsubmittedWork()
        {
            return m_operationsRecordedInCurrentCommandList;
        }

        // Forces the descriptor heap to be reset to D3D before executing future operations
        void InvalidateDescriptorHeap()
        {
            m_currentDescriptorHeap = nullptr;
        }

    private:
        struct CommandAllocatorInfo
        {
            ComPtr<ID3D12CommandAllocator> allocator;

            // The event which will be signaled when the last command list submitted using this allocator
            // completes execution on the GPU.
            GpuEvent completionEvent = {};

            ID3D12CommandAllocator* Get() const { return allocator.Get(); }
        };

        void CloseAndExecute(_In_opt_ ID3D12GraphicsCommandList* commandList);

        std::shared_ptr<CommandQueue> m_queue;
        ComPtr<ID3D12Device> m_d3dDevice;
        ComPtr<IDMLDevice> m_dmlDevice;
        ComPtr<IDMLOperatorInitializer> m_initializer;
        ComPtr<IDMLCommandRecorder> m_recorder;

        // Descriptors are allocated from a pool. The current heap pointer is only used to avoid redundantly
        // setting the same heap; it does not have ownership of the heap object.
        DescriptorPool m_descriptorPool;
        ID3D12DescriptorHeap* m_currentDescriptorHeap = nullptr;

        // The weak pointer avoids a circular reference from context->recorder->allocator->context
        std::shared_ptr<DmlGpuAllocator> m_allocator;

        // The command list currently being recorded into, and whether any command have been recorded yet.
        ComPtr<ID3D12GraphicsCommandList> m_currentCommandList;
        bool m_operationsRecordedInCurrentCommandList = false;

        static constexpr int commandListCount = 3;

        // We use enough command lists and allocators to allow command lists to be reset in a different thread while
        // there is another command list ready to receive commands. When we execute and close a command list, we start
        // the resetting process on a different thread and set m_currentCommandList to the next available one.
        std::array<ComPtr<ID3D12GraphicsCommandList>, commandListCount> m_commandListRing;
        std::array<CommandAllocatorInfo, commandListCount> m_allocatorRing;

        // We should always have 1 less reset thread than command lists since we always need a clean command list, but
        // the other ones can all be in the process of getting reset
        std::array<std::unique_ptr<std::thread>, commandListCount - 1> m_resetThreads;
        Optional<DmlBuffer> m_temporaryBuffer;

        void SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap);
    };

} // namespace Dml
