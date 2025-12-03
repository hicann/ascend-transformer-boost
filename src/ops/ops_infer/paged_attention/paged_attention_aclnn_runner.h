/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_PAGED_ATTENTION_ACLNN_RUNNER_H
#define ATB_PAGED_ATTENTION_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc = aclnnStatus (*)(
    const aclTensor *, const aclTensorList *, const aclTensorList *, const aclTensor *, const aclTensor *,
    const aclIntArray *, const aclIntArray *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclIntArray *, const aclTensor *, const aclTensor *, const aclTensor *, int64_t, double,
    int64_t, int64_t, char *, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, const aclTensor *,
    const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnFusedInferAttentionScoreV3Func = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

namespace atb {
struct AclnnFusedInferAttentionScoreV3Param {
    int64_t numHeads = 0;
    double scaleValue = 0.0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    std::string inputLayoutStr = "";
    int64_t numKeyValueHeads = 0;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    int64_t blockSize = 0;
    int64_t antiquantMode = 0;
    bool softmaxLseFlag = false;
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
};

class PagedAttentionAclnnRunner : public AclnnRunner {
public:
    explicit PagedAttentionAclnnRunner(const infer::PagedAttentionParam &param);
    ~PagedAttentionAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    void InitAclnnParam();
    Status CreateQueryAclnnTensor();
    Status CreateKeyAclnnTensorList();
    Status CreateValueAclnnTensorList();
    Status CreateBlockTableAclnnTensor();
    Status CreateActualSeqLengthKvAclIntArray();
    Status CreateKeyAntiquantScaleAclnnTensor();
    Status CreateValueAntiquantScaleAclnnTensor();
    Status CreateAttentionOutAclnnTensor();
    std::shared_ptr<AclNNTensor> InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex);

private:
    infer::PagedAttentionParam param_;

    size_t aclInTensorNum_ = 0;
    size_t aclOutTensorNum_ = 0;
    size_t aclInTensorListNum_ = 0;
    size_t aclOutTensorListNum_ = 0;

    size_t atbInTensorIndex_ = 0;
    size_t aclInTensorIndex_ = 0;
    size_t aclInTensorListIndex_ = 0;
    size_t atbOutTensorIndex_ = 0;
    size_t aclOutTensorIndex_ = 0;
    size_t aclOutTensorListIndex_ = 0;

    size_t queryAclTensorIndex_ = 0;
    size_t keyAclTensorListIndex_ = 0;
    size_t valueAclTensorListIndex_ = 0;
    size_t blockTableAclTensorIndex_ = 0;
    size_t keyAntiquantScaleAclTensorIndex_ = 0;
    size_t valueAntiquantScaleAclTensorIndex_ = 0;
    size_t attentionOutAclTensorIndex_ = 0;

    aclIntArray *actualSeqLengthsKv_ = nullptr;

    AclnnFusedInferAttentionScoreV3Param aclnnParam_;

    static AclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc aclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc_;
    static AclnnFusedInferAttentionScoreV3Func aclnnFusedInferAttentionScoreV3Func_;
};
} // namespace atb
#endif // ATB_PAGED_ATTENTION_ACLNN_RUNNER_H