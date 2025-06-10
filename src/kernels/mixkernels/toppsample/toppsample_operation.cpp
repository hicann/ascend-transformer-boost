/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <algorithm>
#include <mki_loader/op_register.h>
#include <mki/base/operation_base.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
#include "sink_common.h"

namespace AtbOps {
using namespace Mki;
class ToppsampleOperation : public OperationBase {
public:
    explicit ToppsampleOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("ToppsampleKernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Toppsample), "OpParam is invalid", return 0);
        return DIM_1; // 2 inputs
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Toppsample), "OpParam is invalid", return 0);
        return DIM_2;  // select_index, select_range
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        for (auto &t: outTensors) {
            Mki::TensorDesc desc;
            desc.format = Mki::TENSOR_FORMAT_ND;
            t.desc = desc;
        }
        return opInferShape::CallGeInferShape("TopPSample", launchParam, outTensors,
                                              AsdOps::GetMkiSpecificAttr<OpParam::Toppsample>);
    }
};

REG_OPERATION(ToppsampleOperation);
} //    namespace AtbOps
