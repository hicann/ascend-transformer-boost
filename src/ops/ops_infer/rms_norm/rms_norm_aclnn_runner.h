/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_RMS_NORM_ACLNN_RUNNER_H
#define ATB_RMS_NORM_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnRmsNormGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *x, const aclTensor *gamma, double epsilon,
    const aclTensor *yOut, const aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);

using AclnnRmsNormFunc = aclnnStatus (*)(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

namespace atb {
class RmsNormAclnnRunner : public AclnnRunner {
public:
    explicit RmsNormAclnnRunner(const infer::RmsNormParam &param);
    ~RmsNormAclnnRunner() override;

    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    Status CreateXAclnnTensor();
    Status CreateGammaAclnnTensor();
    Status CreateYOutAclnnTensor();
    Status CreateRstdOutAclnnTensor();

private:
    infer::RmsNormParam param_;

    size_t aclInTensorNum_ = 0;
    size_t aclOutTensorNum_ = 0;

    size_t atbInTensorIndex_ = 0;
    size_t aclInTensorIndex_ = 0;
    size_t atbOutTensorIndex_ = 0;
    size_t aclOutTensorIndex_ = 0;

    size_t xAclTensorIndex_ = 0;
    size_t gammaAclTensorIndex_ = 0;
    size_t yOutAclTensorIndex_ = 0;
    size_t rstdOutAclTensorIndex_ = 0;

    static AclnnRmsNormGetWorkspaceSizeFunc aclnnRmsNormGetWorkspaceSizeFunc_;
    static AclnnRmsNormFunc aclnnRmsNormFunc_;
};
}  // namespace atb
#endif
