/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_FUSION_OPS_RUNNER_H
#define ATB_FUSION_OPS_RUNNER_H
#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
namespace atb {
class FusionOpsRunner : public OpsRunner {
public:
    explicit FusionOpsRunner(const infer::FusionParam &param);
    ~FusionOpsRunner() override;
private:
    infer::FusionParam param_;
    Mki::Tensor nullTensor_ = {};
    bool SetIntensor(KernelGraphNode &fusionNode);
    void SetOuttensor(KernelGraphNode &fusionNode);
    uint32_t GetIntensorSize() const;
    AtbOps::OpParam::Fusion::FusionType GetOpFusionType() const;
    Mki::TensorDType GetOutTensorType(const aclDataType outType) const;
};
} // namespace atb