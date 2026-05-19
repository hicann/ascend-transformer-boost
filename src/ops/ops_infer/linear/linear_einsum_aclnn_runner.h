/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_EINSUM_ACLNN_RUNNER_H
#define ATB_LINEAR_EINSUM_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnTransposeBatchMatMulGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                                                       const aclTensor *, const aclTensor *,
                                                                       const aclIntArray *, const aclIntArray *,
                                                                       const aclIntArray *, int8_t, const int32_t,
                                                                       aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnTransposeBatchMatMulExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class LinearEinsumAclnnRunner : public AclnnRunner {
public:
    explicit LinearEinsumAclnnRunner(const infer::LinearParam &param);
    ~LinearEinsumAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    Status CreateMatmulSelfAclnnTensor();
    Status CreateMatmulMat2AclnnTensor();
    Status CreateMatmulOutAclnnTensor();
    aclnnStatus SetAclnnTransposeBatchMatMulWorkspaceExecutor();
    aclnnStatus SetAclnnTransposeBatchMatMulWeightNzWorkspaceExecutor();
    Status CreatePermArrays();
    void DestroyPermArrays();
    std::shared_ptr<AclNNTensor> CreateXAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> CreateWeightAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> CreateWeightNzAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> CreateOutputAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex);

private:
    infer::LinearParam param_;

    bool isWeightNz_ = false;

    size_t aclInTensorNum_ = 0;
    size_t aclOutTensorNum_ = 0;

    size_t atbInTensorIndex_ = 0;
    size_t aclInTensorIndex_ = 0;
    size_t atbOutTensorIndex_ = 0;
    size_t aclOutTensorIndex_ = 0;

    size_t matmulSelfAclTensorIndex_ = 0;
    size_t matmulMat2AclTensorIndex_ = 0;
    size_t matmulOutAclTensorIndex_ = 0;

    aclIntArray *permX1_ = nullptr;
    aclIntArray *permX2_ = nullptr;
    aclIntArray *permY_ = nullptr;

    static AclnnTransposeBatchMatMulGetWorkspaceSizeFunc aclnnTransposeBatchMatMulGetWorkspaceSizeFunc_;
    static AclnnTransposeBatchMatMulExecuteFunc aclnnTransposeBatchMatMulExecuteFunc_;
    static AclnnTransposeBatchMatMulGetWorkspaceSizeFunc aclnnTransposeBatchMatMulWeightNzGetWorkspaceSizeFunc_;
    static AclnnTransposeBatchMatMulExecuteFunc aclnnTransposeBatchMatMulWeightNzExecuteFunc_;
};
} // namespace atb
#endif // ATB_LINEAR_EINSUM_ACLNN_RUNNER_H
