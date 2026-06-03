/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_GELU_ACLNN_RUNNER_H
#define ATB_GELU_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnGeluV2GetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *x, int64_t approximate,
                                                         const aclTensor *out, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor);

using AclnnGeluV2Func = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class GeluAclnnRunner : public AclnnRunner {
public:
    explicit GeluAclnnRunner(const infer::ActivationParam &param);
    ~GeluAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::ActivationParam param_;

    static AclnnGeluV2GetWorkspaceSizeFunc aclnnGeluV2GetWorkspaceSizeFunc_;
    static AclnnGeluV2Func aclnnGeluV2Func_;
};
} // namespace atb
#endif
