/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_LINEAR_PARALLEL_ACLNN_RUNNER_H
#define ATB_LINEAR_PARALLEL_ACLNN_RUNNER_H

#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"
#include "atb/runner/hccl_runner.h"

namespace atb {
class LinearParallelAclnnRunner : public AclnnRunner {
public:
    explicit LinearParallelAclnnRunner(const infer::LinearParallelParam &param);
    LinearParallelAclnnRunner(const infer::LinearParallelParam &param, bool useRankTableFile);
    LinearParallelAclnnRunner(const infer::LinearParallelParam &param, HcclComm hcclComm);
    ~LinearParallelAclnnRunner() override;
    static Status LoadMethodMatmulReduceScatter();
    // static Status LoadMethodAllGatherMatmul();
    // static Status LoadMethodAlltoAllAllGatherBatchMatMul();
    // static Status LoadMethodBatchMatMulReduceScatterAlltoAll();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status BuildAclnnVariantPackMatmulReduceScatter(const RunnerVariantPack &runnerVariantPack);
    // Status BuildAclnnVariantPackAllGatherMatmul(const RunnerVariantPack &runnerVariantPack);
    // Status BuildAclnnVariantPackAlltoAllAllGatherBatchMatMul(const RunnerVariantPack &runnerVariantPack);
    // Status BuildAclnnVariantPackBatchMatMulReduceScatterAlltoAll(const RunnerVariantPack &runnerVariantPack);
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    aclnnStatus SetAclNNWorkspaceExecutorMatmulReduceScatter();
    // aclnnStatus SetAclNNWorkspaceExecutorAllGatherMatmul();
    // aclnnStatus SetAclNNWorkspaceExecutorAlltoAllAllGatherBatchMatMul();
    // aclnnStatus SetAclNNWorkspaceExecutorBatchMatMulReduceScatterAlltoAll();
    Status LaunchAclnnKernel() override;
    

private:
    HcclRunner hcclRunner_;
    infer::LinearParallelParam param_;
    static aclnnStatus (*aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, int64_t, const char *, const char *, int64_t, int64_t, int64_t, const char *,
        const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnMatmulReduceScatterV2Func_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // static aclnnStatus (*aclnnAllGatherMatmulV2GetWorkspaceSizeFunc_)(
    //     const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    //     const aclTensor *, int64_t, const char *, int64_t, int64_t, int64_t, int64_t, int64_t, const char *,
    //     const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
    // static aclnnStatus (*aclnnAllGatherMatmulV2Func_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // static aclnnStatus (*aclnnAlltoAllvGroupedMatmulV2GetWorkspaceSizeFunc_)(
    //     const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    //     const aclTensor *, const aclTensor *, const aclTensor *, const char *, int64_t, const aclIntArray *,
    //     const aclIntArray *, bool, bool, bool, int64_t, int64_t, const char *, const aclTensor *, const aclTensor *,
    //     const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
    // static aclnnStatus (*aclnnAlltoAllvGroupedMatmulV2Func_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // static aclnnStatus (*aclnnGroupedMatmulAlltoAllvV2GetWorkspaceSizeFunc_)(
    //     const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    //     const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const char *, const char *, int64_t,
    //     int64_t, int64_t, const aclIntArray *, const aclIntArray *, bool, bool, const aclTensor *, const aclTensor *,
    //     uint64_t *, aclOpExecutor **);
    // static aclnnStatus (*aclnnGroupedMatmulAlltoAllvV2Func_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
} // namespace atb
#endif