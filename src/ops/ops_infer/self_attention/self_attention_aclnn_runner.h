/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELF_ATTENTION_ACLNN_RUNNER_H
#define SELF_ATTENTION_ACLNN_RUNNER_H

#include <string>
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc = aclnnStatus (*)(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, const aclTensor *attentionOut, const aclTensor *softmaxLse,
    uint64_t *workspaceSize, aclOpExecutor **executor);
using AclnnFusedInferAttentionScoreV3Func = aclnnStatus (*)(void *workspace, uint64_t workspaceSize,
                                                           aclOpExecutor *executor, const aclrtStream stream);


namespace atb {
struct AclnnSelfAttentionParam {
    int64_t numHeads = 0;
    double scaleValue = 0.0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    std::string inputLayout = "TND";
    int64_t numKeyValueHeads = 0;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 0;
    int64_t blockSize = 0;
    int64_t antiquantMode = 0;
    bool softmaxLseFlag = false;
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
};

class SelfAttentionAclnnRunner : public AclnnRunner {
public:
    explicit SelfAttentionAclnnRunner(const atb::infer::SelfAttentionParam &opParam);
    ~SelfAttentionAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    uint32_t GetInputNum() const;
    uint32_t GetInputNumAclnn() const;
    uint32_t GetOutputNumAclnn() const;
    uint32_t GetInputNumAtb() const;
    uint32_t GetOutputNumAtb() const;
    uint32_t GetOutputNum() const;
    atb::Status CreateAclnnInTensor(const RunnerVariantPack &runnerVariantPack);
    atb::Status CreateAclnnInTensorList(const RunnerVariantPack &runnerVariantPack);
    atb::Status CreateAclnnOutTensor(const RunnerVariantPack &runnerVariantPack);
    atb::Status ParamTransfer(AclnnSelfAttentionParam &targetParam);
    atb::Status ProcessSeqLengthTensor(const atb::Tensor tensor);
    infer::SelfAttentionParam param_;
    int64_t q_S_ = 0;
    int64_t mask_S_ = 0;
    std::vector<int> seqLencache_{};
    std::vector<int64_t> seqLen_{};
    aclIntArray *actualSeqLengths_ = nullptr;
    static AclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc aclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc_;
    static AclnnFusedInferAttentionScoreV3Func aclnnFusedInferAttentionScoreV3Func_;
};
} // namespace atb

#endif