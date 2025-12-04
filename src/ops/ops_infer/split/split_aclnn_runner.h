/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SPLIT_ACLNN_RUNNER_H
#define ATB_SPLIT_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using SplitTensorFuncType = aclnnStatus (*)(const aclTensor *, uint64_t, int64_t, aclTensorList *, uint64_t *,
                                            aclOpExecutor **);
using SplitWithSizeFuncType = aclnnStatus (*)(const aclTensor *, aclIntArray *, int64_t, aclTensorList *, uint64_t *,
                                              aclOpExecutor **);
using ExecuteFuncType = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class SplitAclnnRunner : public AclnnRunner {
public:
    explicit SplitAclnnRunner(const infer::SplitParam &param);
    ~SplitAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;
    void AssignFunc();

private:
    infer::SplitParam param_;
    bool splitWithSize_ = false;
    aclIntArray *splitSizeArray_ = nullptr;

    static SplitTensorFuncType splitGetWorkspaceSizeFunc_;
    static ExecuteFuncType splitExecuteFunc_;
    static SplitWithSizeFuncType splitWithSizeGetWorkspaceSizeFunc_;
    static ExecuteFuncType splitWithSizeExecuteFunc_;

    ExecuteFuncType executeFunc_ = nullptr;
};
} // namespace atb
#endif // ATB_SPLIT_ACLNN_RUNNER_H
