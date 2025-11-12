/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_ACLNN_RUNNER_H
#define ATB_LINEAR_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
class LinearAclnnRunner : public AclnnRunner {
public:
    explicit LinearAclnnRunner(const infer::LinearParam &param);
    ~LinearAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    void GetInTensorNum();
    Status CreateXAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex);
    Status CreateWeightAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex);
    Status CreateBiasAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex);
    Status CreateDeqScaleAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex);
    Status CreateOutputAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> InitAclnnTensor(Tensor atbTensor, size_t aclnnTensorIndex);

private:
    infer::LinearParam param_;
    size_t inTensorNum_ = 2;
    size_t outTensorNum_ = 1;
    aclScalar *alpha_ = nullptr;
    aclScalar *beta_ = nullptr;

    static aclnnStatus (*aclnnMatmulGetWorkspaceSizeFunc_)(
        const aclTensor *, const aclTensor *, const aclTensor *, int8_t, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnMatmulExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    static aclnnStatus (*aclnnAddmmGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, const aclTensor *,
        aclScalar *, aclScalar *, const aclTensor *, int8_t, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnAddmmExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
}  // namespace atb
#endif  // ATB_LINEAR_ACLNN_RUNNER_H