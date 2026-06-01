/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_DEQUANT_ACLNN_RUNNER_H
#define ATB_LINEAR_DEQUANT_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnQuantMatmulV5GetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *,
    bool, bool, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnQuantMatmulV5ExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AclnnQuantMatmulWeightNzGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *,
    bool, bool, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnQuantMatmulWeightNzExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class LinearDequantAclnnRunner : public AclnnRunner {
public:
    explicit LinearDequantAclnnRunner(const infer::LinearParam &param);
    ~LinearDequantAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    Status CreateXAclnnTensor();
    Status CreateWeightAclnnTensor();
    Status CreateDeqScaleAclnnTensor();
    Status CreateBiasAclnnTensor();
    Status CreatePerTokenScaleAclnnTensor();
    Status CreateOutAclnnTensor();
    aclnnStatus SetAclnnQuantMatmulWorkspaceExecutor();
    aclnnStatus SetAclnnQuantMatmulWeightNzWorkspaceExecutor();
    std::shared_ptr<AclNNTensor> InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex);

    infer::LinearParam param_;
    bool isWeightNz_ = false;

    size_t aclInTensorNum_ = 0;
    size_t atbInTensorIndex_ = 0;
    size_t aclInTensorIndex_ = 0;

    size_t xAclTensorIndex_ = 0;
    size_t weightAclTensorIndex_ = 0;
    size_t descaleAclTensorIndex_ = 0;
    size_t biasAclTensorIndex_ = 0;
    size_t perTokenScaleAclTensorIndex_ = 0;
    size_t outAclTensorIndex_ = 0;

    static AclnnQuantMatmulV5GetWorkspaceSizeFunc aclnnQuantMatmulV5GetWorkspaceSizeFunc_;
    static AclnnQuantMatmulV5ExecuteFunc aclnnQuantMatmulV5ExecuteFunc_;
    static AclnnQuantMatmulWeightNzGetWorkspaceSizeFunc aclnnQuantMatmulWeightNzGetWorkspaceSizeFunc_;
    static AclnnQuantMatmulWeightNzExecuteFunc aclnnQuantMatmulWeightNzExecuteFunc_;
};
} // namespace atb
#endif
