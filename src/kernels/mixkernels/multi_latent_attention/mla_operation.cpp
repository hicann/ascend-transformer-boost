/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include <mki/utils/checktensor/check_tensor.h>
#include "atbops/params/params.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"

namespace AtbOps {
using namespace Mki;
class MLAOperation : public OperationBase {
public:
    explicit MLAOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MLA),
            "OpParam is invalid", return nullptr);
        auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::MLA::SPLIT_CACHE:
                return GetKernelByName("MLAKernel");
            default:
                break;
        }
        MKI_LOG(ERROR) << "Unsupport MLA type " << param.type;
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::MLA), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::MLA>(specificParam);
        switch (param.type) {
            case OpParam::MLA::SPLIT_CACHE:
                return DIM_8;
            default:
                break;
        }
        return DIM_1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::MLA), "OpParam is invalid", return 0);
        return DIM_2;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MLA),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::MLA::SPLIT_CACHE:
                return InferShapeMLA(launchParam, outTensors);
            default:
                break;
        }
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Param is invalid");
    }

private:
    Status InferShapeMLA(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
        MKI_CHECK(CheckMLA(launchParam), "Failed to check launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorQRope = launchParam.GetInTensor(DIM_1);
        outTensors[DIM_0].desc = tensorQ.desc;
        outTensors[DIM_0].desc.dtype = tensorQRope.desc.dtype;
        if (param.isRing) {
            // outTensor1  lse
            outTensors[DIM_1].desc = tensorQ.desc;
            if (tensorQ.desc.dims.size() == DIM_2) {
                outTensors[DIM_1].desc.dims[DIM_1] = DIM_1;
            } else if (tensorQ.desc.dims.size() == DIM_3) {
                outTensors[DIM_1].desc.dims[DIM_2] = DIM_1;
            } else {
                outTensors[DIM_1].desc.dims[DIM_0] = DIM_0;
            }
        } else {
            outTensors[DIM_1].desc.dtype = tensorQ.desc.dtype;
            outTensors[DIM_1].desc.format = tensorQ.desc.format;
            outTensors[DIM_1].desc.dims = {0};
        }
        return Status::OkStatus();
    }

    bool CheckMLA(const LaunchParam &launchParam) const
    {
        return true;
    }
};

REG_OPERATION(MLAOperation);
} // namespace AtbOps
