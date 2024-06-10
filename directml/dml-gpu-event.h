#pragma once

#include <memory>
#include <d3d12.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

struct GpuEvent
{
    uint64_t fenceValue;
    ComPtr<ID3D12Fence> fence;

    bool IsSignaled() const
    {
        return fence->GetCompletedValue() >= fenceValue;
    }

    // Blocks until IsSignaled returns true.
    void WaitForSignal() const {
        if (IsSignaled()) {
        return;  // early-out
        }

        while (!IsSignaled()) {
    #if defined(_M_AMD64) || defined(__x86_64__)
        _mm_pause();
    #endif
        }
    }
};