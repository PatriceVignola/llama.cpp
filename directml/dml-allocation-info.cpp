// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dml-allocation-info.h"
#include "dml-reserved-resource-sub-allocator.h"
#include "dml-sub-allocator.h"

namespace Dml
{

    AllocationInfo::~AllocationInfo()
    {
        if (m_owner)
        {
            m_owner->FreeResource(this, m_pooledResourceId);
        }
    }

} // namespace Dml