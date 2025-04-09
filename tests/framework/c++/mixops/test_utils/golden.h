/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef MIXOPS_TEST_UTILS_GOLDEN_H
#define MIXOPS_TEST_UTILS_GOLDEN_H
#include "atb_mixops_test.h"
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/status/status.h>
#include "float_util.h"

namespace AtbOps {
class Golden {
public:
    static Status InOutTensorEqual(float atol, float rtol, const GoldenContext &context)
    {
        const Mki::Tensor &inTensor = context.hostInTensors.at(0);
        const Mki::Tensor &outTensor = context.hostOutTensors.at(0);
        for (int64_t i = 0; i < inTensor.Numel(); i++) {
            float expect = inTensor.desc.dtype == TENSOR_DTYPE_FLOAT16
                            ? static_cast<float>(static_cast<fp16_t *>(inTensor.data)[i])
                            : static_cast<float *>(inTensor.data)[i];
            float result = outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16
                            ? static_cast<float>(static_cast<fp16_t *>(outTensor.data)[i])
                            : static_cast<float *>(outTensor.data)[i];
            if (!FloatUtil::FloatJudgeEqual(expect, result, atol, rtol)) {
                std::string msg = "pos " + std::to_string(i) + ", expect: " + std::to_string(expect) +
                                ", result: " + std::to_string(result);
                return Mki::Status::FailStatus(-1, msg);
            }
        }
        return Mki::Status::OkStatus();
    }
};
}

#endif