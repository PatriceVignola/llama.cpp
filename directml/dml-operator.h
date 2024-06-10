#pragma once

#include "DirectMLX.h"
#include "dml-buffer-region.h"

class DmlOperator {
public:
    virtual void RecordDispatch(
        ID3D12GraphicsCommandList* command_list,
        const Dml::D3D12BufferRegion& temporary_buffer_region) = 0;
    virtual void UpdateBindings(
        ID3D12Device* d3d12Device,
        void** raw_input_data,
        void* raw_output_data,
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions) = 0;
    virtual uint64_t GetTemporaryResourceSize() const = 0;
    virtual ~DmlOperator() {}
    virtual const void* GetRawInputData(int index) const = 0;
    virtual const void* GetRawOutputData() const = 0;
    virtual bool LateBindingAllowed() const = 0;
};