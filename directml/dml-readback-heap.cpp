// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define NOMINMAX

#include <assert.h>
#include "dml-readback-heap.h"
#include "dml-execution-context.h"

namespace Dml
{
    static ComPtr<ID3D12Resource> CreateReadbackHeap(ID3D12Device* device, size_t size)
    {
        ComPtr<ID3D12Resource> readbackHeap;
        auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size);

        THROW_IF_FAILED(device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(readbackHeap.ReleaseAndGetAddressOf())));

        return readbackHeap;
    }

    ReadbackHeap::ReadbackHeap(ID3D12Device* device) : m_device(device)
    {
    }

    static size_t ComputeNewCapacity(size_t existingCapacity, size_t desiredCapacity)
    {
        size_t newCapacity = existingCapacity;

        while (newCapacity < desiredCapacity)
        {
            if (newCapacity >= std::numeric_limits<size_t>::max() / 2)
            {
                // Overflow; there's no way we can satisfy this allocation request
                THROW_HR(E_OUTOFMEMORY);
            }

            newCapacity *= 2; // geometric growth
        }

        return newCapacity;
    }

    void ReadbackHeap::EnsureReadbackHeap(size_t size)
    {
        if (!m_readbackHeap)
        {
            // Initialize the readback heap for the first time
            assert(m_capacity == 0);
            m_capacity = ComputeNewCapacity(c_initialCapacity, size);
            m_readbackHeap = CreateReadbackHeap(m_device.Get(), m_capacity);
        }
        else if (m_capacity < size)
        {
            // Ensure there's sufficient capacity
            m_capacity = ComputeNewCapacity(m_capacity, size);

            m_readbackHeap = nullptr;
            m_readbackHeap = CreateReadbackHeap(m_device.Get(), m_capacity);
        }

        assert(m_readbackHeap->GetDesc().Width >= size);
    }

    void ReadbackHeap::ReadbackFromGpu(
        ExecutionContext* executionContext,
        uint8_t* dst,
        uint64_t size,
        ID3D12Resource* src,
        uint64_t srcOffset,
        D3D12_RESOURCE_STATES srcState)
    {
        assert(size != 0);

        EnsureReadbackHeap(size);

        // Copy from the source resource into the readback heap
        executionContext->CopyBufferRegion(
            m_readbackHeap.Get(),
            0,
            D3D12_RESOURCE_STATE_COPY_DEST,
            src,
            srcOffset,
            srcState,
            size);

        // Wait for completion and map the result
        executionContext->Flush();
        executionContext->GetCurrentCompletionEvent().WaitForSignal();
        executionContext->ReleaseCompletedReferences();

        // Map the readback heap and copy it into the destination
        void* readbackHeapData = nullptr;
        THROW_IF_FAILED(m_readbackHeap->Map(0, nullptr, &readbackHeapData));
        memcpy(dst, readbackHeapData, size);
        m_readbackHeap->Unmap(0, nullptr);
    }

    void ReadbackHeap::ReadbackFromGpu(
        ExecutionContext* executionContext,
        std::vector<uint8_t*>& dst,
        const std::vector<uint32_t>& dstSizes,
        std::vector<ID3D12Resource*>& src,
        D3D12_RESOURCE_STATES srcState)
    {
        assert(dst.size() == src.size());
        assert(dstSizes.size() == src.size());

        if (dst.empty())
        {
            return;
        }

        uint32_t totalSize = 0;
        for (auto size : dstSizes)
        {
            totalSize += size;
        }

        EnsureReadbackHeap(totalSize);

        // Copy from the source resource into the readback heap
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dst.size(); ++i)
        {
            executionContext->CopyBufferRegion(
                m_readbackHeap.Get(),
                offset,
                D3D12_RESOURCE_STATE_COPY_DEST,
                src[i],
                0,
                srcState,
                dstSizes[i]);

            offset += dstSizes[i];
        }

        // Wait for completion and map the result
        executionContext->Flush();
        executionContext->GetCurrentCompletionEvent().WaitForSignal();
        executionContext->ReleaseCompletedReferences();

        // Map the readback heap and copy it into the destination
        void* readbackHeapData = nullptr;
        THROW_IF_FAILED(m_readbackHeap->Map(0, nullptr, &readbackHeapData));

        // Copy from the source resource into the readback heap
        offset = 0;
        for (uint32_t i = 0; i < dst.size(); ++i)
        {
            memcpy(dst[i], static_cast<uint8_t*>(readbackHeapData) + offset, dstSizes[i]);
            offset += dstSizes[i];
        }

        m_readbackHeap->Unmap(0, nullptr);
    }
} // namespace Dml
