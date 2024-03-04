// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "dml-gpu-allocator.h"
#include "dml-reserved-resource-sub-allocator.h"
#include "dml-tagged-pointer.h"
#include "dml-allocation-info.h"
#include "dml-allocator-rounding-mode.h"

namespace Dml
{
    DmlGpuAllocator::DmlGpuAllocator(std::shared_ptr<DmlReservedResourceSubAllocator> bfcSubAllocator)
        : m_bfcSubAllocator(bfcSubAllocator) {}

    void* DmlGpuAllocator::Alloc(size_t sizeInBytes)
    {
        return Alloc(sizeInBytes, m_defaultRoundingMode);
    }

    void* DmlGpuAllocator::Alloc(size_t sizeInBytes, AllocatorRoundingMode roundingMode)
    {
        return m_bfcSubAllocator->Alloc(sizeInBytes);
    }

    void DmlGpuAllocator::Free(void* ptr)
    {
        m_bfcSubAllocator->Free(ptr);
    }

    D3D12BufferRegion DmlGpuAllocator::CreateBufferRegion(const void* opaquePointer, uint64_t sizeInBytes)
    {
        return m_bfcSubAllocator->CreateBufferRegion(opaquePointer, sizeInBytes);
    }

    AllocationInfo* DmlGpuAllocator::GetAllocationInfo(void* opaquePointer)
    {
        return m_bfcSubAllocator->GetAllocationInfo(opaquePointer);
    }

    void DmlGpuAllocator::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
    {
        m_defaultRoundingMode = roundingMode;
    }

    DmlBuffer DmlGpuAllocator::AllocateDefaultBuffer(uint64_t num_bytes)
    {
        return DmlBuffer(this, num_bytes, m_defaultRoundingMode);
    }

    DmlBuffer DmlGpuAllocator::AllocateDefaultBuffer(uint64_t num_bytes, AllocatorRoundingMode roundingMode)
    {
        return DmlBuffer(this, num_bytes, roundingMode);
    }

    uint64_t DmlGpuAllocator::GetUniqueId(void* opaquePointer)
    {
        return m_bfcSubAllocator->GetUniqueId(opaquePointer);
    }

} // namespace Dml