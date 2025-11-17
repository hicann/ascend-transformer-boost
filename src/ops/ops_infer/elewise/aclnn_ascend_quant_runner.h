/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ACLNN_ASCEND_QUANT_RUNNER_H
#define ATB_ACLNN_ASCEND_QUANT_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"


using AclnnAscendQuantGetWorkspaceFunc = aclnnStatus(*)(
    const aclTensor *, const aclTensor *, const aclTensor *, bool, char*,
    int32_t, int32_t, const aclTensor *, uint64_t *, aclOpExecutor **);

using AclnnAscendQuantExecuteFunc = aclnnStatus(*)(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);

namespace atb {
class AclnnAscendQuantRunner : public AclnnRunner {
public:
    explicit AclnnAscendQuantRunner(const infer::ElewiseParam &param);
    ~AclnnAscendQuantRunner() override;

    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::ElewiseParam param_;

    // 对应aclnnop/aclnn_ascend_quant.h中的两段式接口
    static AclnnAscendQuantGetWorkspaceFunc aclnnGetWorkspaceSizeFunc_;
    static AclnnAscendQuantExecuteFunc aclnnExecuteFunc_;
};
} // namespace atb
#endif
