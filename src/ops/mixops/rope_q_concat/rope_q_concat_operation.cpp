/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include "atbops/params/params.h"

namespace AtbOps {
using namespace Mki;
class RopeQConcatOperation : public OperationBase {
public:
    explicit RopeQConcatOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RopeQConcat), "OpParam is invalid", return 0);
        return 4; // RopeQConcat Op has 4 inputs: q, k, cos, sin
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RopeQConcat), "OpParam is invalid", return 0);
        return 1; // RopeQConcat Op has 1 outpts: ropeQConcat
    }

    bool CheckRopeQConcat(const LaunchParam &launchParam) const
    {
        auto &inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.dims.size() == 2, "dim size of inTensor0 is invalid", return false);
        auto &inTensor1 = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(inTensor1.desc.dims.size() == 2, "dim size of inTensor1 is invalid", return false);
        auto &inTensor2 = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(inTensor2.desc.dims.size() == 2, "dim size of inTensor2 is invalid", return false);
        auto &inTensor3 = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(inTensor3.desc.dims.size() == 3, "dim size of inTensor3 is invalid", return false);

        MKI_CHECK(inTensor1.desc.dims.size() == inTensor2.desc.dims.size(), "dim size of cos and sin is not equal",
                     return false);
        MKI_CHECK(inTensor1.desc.dims[0] == inTensor2.desc.dims[0] &&
            inTensor1.desc.dims[DIM_1] == inTensor2.desc.dims[DIM_1],
            "Shape of cos/sin should be same", return false);
        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckRopeQConcat(launchParam), "Failed to check run info",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto &inTensor1 = launchParam.GetInTensor(DIM_1);
        auto &inTensor3 = launchParam.GetInTensor(DIM_3);
        outTensors[DIM_0].desc = launchParam.GetInTensor(DIM_3).desc;
        outTensors[DIM_0].desc.dims[DIM_2] = inTensor1.desc.dims[DIM_1] + inTensor3.desc.dims[DIM_2];

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("RopeQConcatKernel");
    }
};

REG_OPERATION(RopeQConcatOperation);
} // namespace AtbOps
