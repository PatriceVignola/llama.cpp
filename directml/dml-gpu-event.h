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
    void WaitForSignal() const
    {
        if (IsSignaled())
            return; // early-out

        // wil::unique_handle h(CreateEvent(nullptr, TRUE, FALSE, nullptr));
        // THROW_LAST_ERROR_IF(!h);

        // THROW_IF_FAILED(fence->SetEventOnCompletion(fenceValue, h.get()));

        while (!IsSignaled())
        {
            // DO nothing
        }

        // WaitForSingleObject(h.get(), INFINITE);
    }
};