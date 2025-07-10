/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



#ifndef NORM_ATTR_H
#define NORM_ATTR_H

#include <cstdint>
#include <cstddef>

static const int64_t GE_RMSNORMATB_EPSILON = 0;
static const int64_t GE_RMSNORMATB_HIGN_PRECISION_MODE = 1;
static const int64_t GE_RMSNORMATB_GEMMA_MODE = 2;

static const int64_t GE_POSTLAYERNORM_EPSILON = 0;
static const int64_t GE_POSTLAYERNORM_ZOOMSCALEVALUE = 1;

namespace AsdOps {
const uint8_t *GetMkiSpecificAttrRmsNormAtb(void *attrs, size_t index, uint64_t &type);
const uint8_t *GetMkiSpecificAttrPostLayerNorm(void *attrs, size_t index, uint64_t &type);

} // namespace AsdOps

#endif