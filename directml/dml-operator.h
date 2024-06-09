#pragma once

#include "DirectMLX.h"
#include "dml-buffer-region.h"

class DmlOperator {
public:
    virtual void RecordDispatch(
        ID3D12GraphicsCommandList* command_list,
        const std::vector<Dml::D3D12BufferRegion>& input_buffer_regions,
        const std::vector<Dml::D3D12BufferRegion>& output_buffer_regions) = 0;

    virtual ~DmlOperator() {}
};