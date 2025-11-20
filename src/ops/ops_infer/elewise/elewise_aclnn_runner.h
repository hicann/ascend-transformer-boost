/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ELEWISE_ACLNN_RUNNER_H
#define ATB_ELEWISE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
class ElewiseAclnnRunner : public AclnnRunner {
public:
    explicit ElewiseAclnnRunner(const infer::ElewiseParam &param);
    ~ElewiseAclnnRunner() override;

    static Status LoadMethod(infer::ElewiseParam::ElewiseType elewiseType);

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::ElewiseParam param_;
    uint32_t inTensorNum_ = 0;
    float varAttrFloat_ = 1.0f;
    int32_t varAttrInt = 1;
    aclScalar *alpha_ = nullptr;
    // 对应aclnnop/aclnn_cast.h中的两段式接口
    static aclnnStatus (*aclnnCastGetWorkspaceSizeFunc_)(const aclTensor *, const aclDataType, const aclTensor *,
                                                         uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnCastExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_cos.h中的两段式接口
    static aclnnStatus (*aclnnCosGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                        aclOpExecutor **);
    static aclnnStatus (*aclnnCosExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_sin.h中的两段式接口
    static aclnnStatus (*aclnnSinGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                        aclOpExecutor **);
    static aclnnStatus (*aclnnSinExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_neg.h中的两段式接口
    static aclnnStatus (*aclnnNegGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                        aclOpExecutor **);
    static aclnnStatus (*aclnnNegExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_logical_not.h中的两段式接口
    static aclnnStatus (*aclnnLogicalNotGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                               aclOpExecutor **);
    static aclnnStatus (*aclnnLogicalNotExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_logical_and.h中的两段式接口
    static aclnnStatus (*aclnnLogicalAndGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, const aclTensor *,
                                                               uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnLogicalAndExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_logical_or.h中的两段式接口
    static aclnnStatus (*aclnnLogicalOrGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, const aclTensor *,
                                                              uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnLogicalOrExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_sub.h中的两段式接口
    static aclnnStatus (*aclnnSubGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, const aclScalar *,
                                                        const aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnSubExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_eq_tensor.h中的两段式接口
    static aclnnStatus (*aclnnEqTensorGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, const aclTensor *,
                                                             uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnEqTensorExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_tanh.h中的两段式接口
    static aclnnStatus (*aclnnTanhGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                         aclOpExecutor **);
    static aclnnStatus (*aclnnTanhExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
} // namespace atb
#endif
