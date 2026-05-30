/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_SWIGLU_QUANT_ACLNN_RUNNER_H
#define ATB_SWIGLU_QUANT_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnSwiGluQuantV2GetWorkspaceSizeFunc =
    aclnnStatus (*)(const aclTensor *x, const aclTensor *smoothScalesOptional, const aclTensor *offsetsOptional,
                    const aclTensor *groupIndexOptional, bool activateLeft, char *quantModeOptional,
                    int64_t groupListType, int64_t dstType, const aclTensor *yOut, const aclTensor *scaleOut,
                    uint64_t *workspaceSize, aclOpExecutor **executor);

using AclnnSwiGluQuantV2Func = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnInplaceReciprocalGetWorkspaceSizeFunc =
    aclnnStatus (*)(const aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor);

using AclnnInplaceReciprocalFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class SwigluQuantAclnnRunner : public AclnnRunner {
public:
    explicit SwigluQuantAclnnRunner(const infer::SwigluQuantParam &param);
    ~SwigluQuantAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    bool useCache() override;

private:
    void FreeSmoothScales();
    infer::SwigluQuantParam param_;
    std::shared_ptr<AclNNTensor> smoothScalesAclnnTensor_;
    void *smoothScalesDeviceAddr_ = nullptr;
    uint64_t swigluQuantWorkspaceSize_ = 0;
    uint64_t inplaceReciprocalWorkspaceSize_ = 0;
    std::shared_ptr<aclOpExecutor> aclnnInplaceReciprocalExecutor_;

    static AclnnSwiGluQuantV2GetWorkspaceSizeFunc aclnnSwiGluQuantV2GetWorkspaceSizeFunc_;
    static AclnnSwiGluQuantV2Func aclnnSwiGluQuantV2Func_;
    static AclnnInplaceReciprocalGetWorkspaceSizeFunc aclnnInplaceReciprocalGetWorkspaceSizeFunc_;
    static AclnnInplaceReciprocalFunc aclnnInplaceReciprocalFunc_;
};
} // namespace atb
#endif
