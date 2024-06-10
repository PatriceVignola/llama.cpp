#pragma once

#include "DirectMLX.h"
#include "dml-buffer-region.h"

class DmlOperator {
public:
    virtual void RecordDispatch(
        ID3D12GraphicsCommandList* command_list,
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions,
        const Dml::D3D12BufferRegion& temporary_buffer_region) = 0;
    virtual uint64_t GetTemporaryResourceSize() = 0;
    virtual ~DmlOperator() {}
};