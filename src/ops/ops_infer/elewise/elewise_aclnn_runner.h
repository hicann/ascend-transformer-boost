/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ELEWISE_ACLNN_RUNNER_H
#define ELEWISE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
using GetWsThunk = std::function<
    aclnnStatus(const AclNNVariantPack &,
                const infer::ElewiseParam &,
                uint64_t * /*workspace size*/,
                aclOpExecutor **)>;

using ExecThunk = std::function<
    aclnnStatus(void * /*workspace*/,
                uint64_t /*workspace size*/,
                aclOpExecutor *,
                aclrtStream)>;

using AclnnAscendQuantGetWsFunc = aclnnStatus(*)(
    const aclTensor *x,
    const aclTensor *scale,
    const aclTensor *offset,
    bool sqrt_mode,
    const char *round_mode,
    int32_t out_dtype,
    int32_t axis,
    const aclTensor *y,
    uint64_t *workspace_size,
    aclOpExecutor **executor);

using AclnnAscendQuantExecFunc = aclnnStatus(*)(
    void *workspace,
    uint64_t workspace_size,
    aclOpExecutor *executor,
    aclrtStream stream);

class ElewiseAclnnRunner : public AclnnRunner {
public:
    explicit ElewiseAclnnRunner(const infer::ElewiseParam &param);
    ~ElewiseAclnnRunner() override;
    static Status LoadAclnnFunctions();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    atb::Tensor scaleTensor = {};
    atb::Tensor offsetTensor = {};

private:
    struct KernelAdapters {
        const char *name;
        GetWsThunk getWs;
        ExecThunk exec;
    };
    static KernelAdapters MakeAdaptersByType(const infer::ElewiseParam &param);
    infer::ElewiseParam param_;
    GetWsThunk adapterGetWs_;
    ExecThunk adapterExec_;

    static AclnnAscendQuantGetWsFunc aclnnAscendQuantGetWorkspaceSizeFunc_;
    static AclnnAscendQuantExecFunc aclnnAscendQuantExecuteFunc_;

};
} // namespace atb

#endif //ELEWISE_ACLNN_RUNNER_H
