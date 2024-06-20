// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "dml-buffer-region.h"
#include "dml-buffer.h"
#include "dml-allocator-rounding-mode.h"

namespace onnxruntime
{
    class BFCArena;
}

namespace Dml
{
    class DmlReservedResourceSubAllocator;
    class BucketizedBufferAllocator;
    class AllocationInfo;
    struct TaggedPointer;

    class DmlGpuAllocator
    {
    public:
        DmlGpuAllocator(std::shared_ptr<DmlReservedResourceSubAllocator> bfcSubAllocator);
        void* Alloc(size_t sizeInBytes);
        void* Alloc(size_t sizeInBytes, AllocatorRoundingMode roundingMode);
        void Free(void* ptr);
        D3D12BufferRegion CreateBufferRegion(const void* opaquePointer, uint64_t sizeInBytes) const;
        AllocationInfo* GetAllocationInfo(void* opaquePointer);
        void SetDefaultRoundingMode(AllocatorRoundingMode roundingMode);
        DmlBuffer AllocateDefaultBuffer(uint64_t num_bytes);
        DmlBuffer AllocateDefaultBuffer(uint64_t num_bytes, AllocatorRoundingMode roundingMode);
        uint64_t GetUniqueId(void* opaquePointer);

    private:
        // This allocator is specific to DML and is used to decode the opaque data returned by the BFC
        // allocator into objects that DML understands
        std::shared_ptr<DmlReservedResourceSubAllocator> m_bfcSubAllocator;

        // Unless specifically requested, allocation sizes are not rounded to enable pooling
        // until SetDefaultRoundingMode is called.  This should be done at completion of session
        // initialization.
        AllocatorRoundingMode m_defaultRoundingMode = AllocatorRoundingMode::Disabled;
    };
} // namespace Dml