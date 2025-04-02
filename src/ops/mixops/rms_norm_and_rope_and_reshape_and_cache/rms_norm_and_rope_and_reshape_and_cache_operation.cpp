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
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "atbops/params/params.h"
 
constexpr int64_t IN_TENSOR_NUM = 7;
constexpr int64_t OUT_TENSOR_NUM = 1;
constexpr uint32_t KEY_CACHE_IN_IDX = 6;
constexpr uint32_t KEY_CACHE_OUT_IDX = 0;
 
namespace AtbOps {
using namespace Mki;
class RmsNormAndRopeAndReshapeAndCacheOperation : public OperationBase {
public:
    explicit RmsNormAndRopeAndReshapeAndCacheOperation(const std::string &opName) noexcept : OperationBase(opName) {}
 
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Fail to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RmsNormAndRopeAndReshapeAndCache),
            "OpParam is invalid", return nullptr);
        return GetKernelByName("RmsNormAndRopeAndReshapeAndCacheKernel");
    }
 
    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return IN_TENSOR_NUM;
    }
 
    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return OUT_TENSOR_NUM;
    }
 
protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RmsNormAndRopeAndReshapeAndCache),
            "no match param type", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        MKI_LOG(INFO) << "RmsNormAndRopeAndReshapeAndCache InferShapeImpl launchParam.GetInTensor().desc="
                      << launchParam.GetInTensor(KEY_CACHE_IN_IDX).desc.ToString();
        outTensors[KEY_CACHE_OUT_IDX].desc = launchParam.GetInTensor(KEY_CACHE_IN_IDX).desc;
 
        return Status::OkStatus();
    }
};
REG_OPERATION(RmsNormAndRopeAndReshapeAndCacheOperation);
} // namespace AtbOps