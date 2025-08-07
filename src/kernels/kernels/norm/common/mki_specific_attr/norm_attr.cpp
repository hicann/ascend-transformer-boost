/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "norm_attr.h"
#include "sink_common.h"
#include "asdops/params/params.h"
#include <cstddef>
#include <mki/utils/log/log.h>
namespace AsdOps {
const uint8_t *GetMkiSpecificAttrRmsNormAtb(void *attrs, size_t index, uint64_t &type)
{
    type = AttrType::BASIC_TYPE;
    const Mki::Any *params = reinterpret_cast<const Mki::Any *>(attrs);
    const AsdOps::OpParam::Norm *normParams = &Mki::AnyCast<AsdOps::OpParam::Norm>(*params);
    switch (index) {
        case GE_RMSNORMATB_EPSILON:
            type |= (sizeof(normParams->epsilon) << AsdOps::SHIFT_BITS);
            return reinterpret_cast<const uint8_t *>(&normParams->epsilon);
        case GE_RMSNORMATB_HIGN_PRECISION_MODE:
            type |= (sizeof(normParams->precisionMode) << AsdOps::SHIFT_BITS);
            return reinterpret_cast<const uint8_t *>(&normParams->precisionMode);
        case GE_RMSNORMATB_GEMMA_MODE:
            type |= (sizeof(normParams->gemmaMode) << AsdOps::SHIFT_BITS);
            return reinterpret_cast<const uint8_t *>(&normParams->gemmaMode);
        default:
            return nullptr;
    }
}
const uint8_t *GetMkiSpecificAttrPostLayerNorm(void *attrs, size_t index, uint64_t &type)
{
    type = AttrType::BASIC_TYPE;
    const Mki::Any *params = reinterpret_cast<const Mki::Any *>(attrs);
    const AsdOps::OpParam::Norm *normParams = &Mki::AnyCast<AsdOps::OpParam::Norm>(*params);
    switch (index) {
        case GE_POSTLAYERNORM_EPSILON:
            type |= (sizeof(normParams->epsilon) << AsdOps::SHIFT_BITS);
            return reinterpret_cast<const uint8_t *>(&normParams->epsilon);
        case GE_POSTLAYERNORM_ZOOMSCALEVALUE:
            type |= (sizeof(normParams->zoomScaleValue) << AsdOps::SHIFT_BITS);
            return reinterpret_cast<const uint8_t *>(&normParams->zoomScaleValue);
        default:
            return nullptr;
    }
}
} // namespace AsdOps