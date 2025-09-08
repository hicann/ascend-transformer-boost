/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_FUSION_OPERATION_H
#define ATB_FUSION_OPERATION_H
#include <acl/acl.h>
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"
namespace atb {
class FusionOperation : public OperationBase {
public:
    explicit FusionOperation(const infer::FusionParam &param);
    ~FusionOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    SVector<bool> GetEmptyInTensorPermissions() const override;
    nlohmann::json GetParamJson() const override;

private:
   infer::FusionParam param_;
//    Status InferShapeCommon(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
};
}
#endif