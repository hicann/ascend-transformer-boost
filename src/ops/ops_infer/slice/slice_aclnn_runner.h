/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SLICE_ACLNN_RUNNER_H
#define ATB_SLICE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnSliceV2GetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclIntArray *, const aclIntArray *,
                                                         const aclIntArray *, const aclIntArray *, aclTensor *,
                                                         uint64_t *, aclOpExecutor **);
using AclnnSliceV2Func = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class SliceAclnnRunner : public AclnnRunner {
public:
    explicit SliceAclnnRunner(const infer::SliceParam &param);
    ~SliceAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    infer::SliceParam param_;
    aclIntArray *stepsArray_ = nullptr;
    aclIntArray *axesArray_ = nullptr;
    aclIntArray *startsArray_ = nullptr;
    aclIntArray *endsArray_ = nullptr;

    static AclnnSliceV2GetWorkspaceSizeFunc aclnnGetWorkspaceSizeFunc_;
    static AclnnSliceV2Func aclnnExecuteFunc_;
};
} // namespace atb
#endif // ATB_SLICE_ACLNN_RUNNER_H
