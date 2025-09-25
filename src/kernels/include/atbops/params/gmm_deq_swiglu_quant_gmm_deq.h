/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_GMM_DEQ_SWIGLU_QUANT_GMM_DEQ_H
#define ATBOPS_PARAMS_GMM_DEQ_SWIGLU_QUANT_GMM_DEQ_H

namespace AtbOps {
namespace OpParam {
struct GmmDeqSwigluQuantGmmDeq {
    enum class OutputType {
        OUTPUT_FLOAT16 = 0,
        OUTPUT_BFLOAT16,
        OUTPUT_INVALID
    };
    using enum OutputType;

    enum class GroupListType {
        GROUP_LIST_CUMSUM = 0,
        GROUP_LIST_SINGLE,
        GROUP_LIST_INVALID
    };
    using enum GroupListType;

    enum class WeightUpPermuteType {
        PERMUTE_N256 = 0,
        PERMUTE_N128,
        PERMUTE_INVALID
    };
    using enum WeightUpPermuteType;

    OutputType outputType = OutputType::OUTPUT_FLOAT16;
    GroupListType groupListType = GroupListType::GROUP_LIST_CUMSUM;
    WeightUpPermuteType weightUpPermuteType = WeightUpPermuteType::PERMUTE_N256;
    bool transposeWeightUp = false;
    bool transposeWeightDown = true;

    bool operator==(const GmmDeqSwigluQuantGmmDeq &other) const
    {
        return outputType == other.outputType && groupListType == other.groupListType &&
            weightUpPermuteType == other.weightUpPermuteType && transposeWeightUp == other.transposeWeightUp &&
            transposeWeightDown == other.transposeWeightDown;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_GMM_DEQ_SWIGLU_QUANT_GMM_DEQ_H