/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mla_preprocess_aclnn_runner.h"

#include "aclnn_mla_preprocess.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include <atbops/params/params.h>

namespace {
// tbd
static const uint32_t IN_TENSOR_NUM = 24;
static const uint32_t OUT_TENSOR_NUM = 4;
static const uint32_t Q_OUT1_INDEX = 2;
static const uint32_t KV_CACHE_OUT1_INDEX = 3;
static const uint32_t KV_CACHE_ROPE_INDEX = 20;
} // namespace

namespace atb {
MlaPreprocessAclnnRunner::MlaPreprocessAclnnRunner(const infer::MlaPreprocessParam &param)
    : AclnnRunner("MlaPreprocessOpsRunner", RUNNER_TYPE_MLA_PREPROCESS_ACLNN), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "MlaPreprocessAclnnRunner::MlaPreprocessAclnnRunner called";
}

MlaPreprocessAclnnRunner::MlaPreprocessAclnnRunner(const infer::MlaPreprocessParam &param, bool doRmsNorm)
    : AclnnRunner("MlaPreprocessOpsRunner", RUNNER_TYPE_MLA_PREPROCESS_ACLNN), param_(param), doRmsNorm_(doRmsNorm)
{
    ATB_LOG(INFO) << GetLogPrefix()
                  << "MlaPreprocessAclnnRunnecr::MlaPreprocessAclnnRunner called, set doRmsNorm: " << doRmsNorm;
}

MlaPreprocessAclnnRunner::~MlaPreprocessAclnnRunner() {}

Status MlaPreprocessAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    bool isRopeCache = param_.cacheMode != infer::MlaPreprocessParam::CacheMode::KVCACHE;
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        if (i == KV_CACHE_ROPE_INDEX && !isRopeCache) {
            // kvCache不带rope转置时kvCacheRope为nullptr
            this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
            continue;
        }
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = i;
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    this->aclnnVariantPack_.aclOutTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        if ((i == Q_OUT1_INDEX || i == KV_CACHE_OUT1_INDEX) && !isRopeCache) {
            // kvCache不带rope转置时不生成2个rope分量
            this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
            continue;
        }
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = i;
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    return atb::NO_ERROR;
}

aclError MlaPreprocessAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn mlaPreprocess setup start.";
    size_t inTensorStart = 0;
    bool isRopeCache = param_.cacheMode != infer::MlaPreprocessParam::CacheMode::KVCACHE;
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn mlaPreprocess isRopeCache: " << isRopeCache
                  << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    aclTensor *input = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *gamma0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *beta0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantScale0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantOffset0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *wdqkv = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *deScale0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *bias0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *gamma1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *beta1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantScale1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantOffset1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *wuq = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *deScale1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *bias1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *gamma2 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *cos = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *sin = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *wuk = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *kvCache = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *kRope = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    if (param_.cacheMode == infer::MlaPreprocessParam::CacheMode::KVCACHE) {
        kRope = nullptr;
    }
    aclTensor *slotmapping = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *ctkvScale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *qNopeScale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    size_t outTensorStart = 0;
    aclTensor *qOut0 = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    aclTensor *kvCacheOut0 = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    aclTensor *qOut1 = nullptr;
    aclTensor *kvCacheOut1 = nullptr;
    if (isRopeCache) {
        qOut1 = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
        kvCacheOut1 = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    }

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    ATB_LOG(FATAL) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                   << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                   << ", raw ptr from it: " << raw_executor_ptr
                   << ", then take the address of the raw ptr: " << &raw_executor_ptr;

    ATB_LOG(FATAL) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    aclError ret = aclnnMlaPreprocessGetWorkspaceSize(
        input, gamma0, beta0, quantScale0, quantOffset0, wdqkv, deScale0, bias0, gamma1, beta1, quantScale1,
        quantOffset1, wuq, deScale1, bias1, gamma2, cos, sin, wuk, kvCache, kRope, slotmapping, ctkvScale, qNopeScale,
        param_.wdqDim, param_.qRopeDim, param_.kRopeDim, param_.epsilon, param_.qRotaryCoeff, param_.kRotaryCoeff,
        param_.transposeWdq, param_.transposeWuq, param_.transposeWuk, param_.cacheMode, param_.quantMode,
        doRmsNorm_, // doRmsNorm
        1,          // wdkvSplitCount
        qOut0, kvCacheOut0, qOut1, kvCacheOut1, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
    bool repeatable = this->executorRepeatable_;
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [repeatable](aclOpExecutor *ptr) {
        if (ptr && repeatable) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

Status MlaPreprocessAclnnRunner::LaunchAclnnKernel(const AclNNVariantPack &aclNNVariantPack)
{
    (void) aclNNVariantPack;
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn mlaPreprocess execute start.";
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclError ret = aclnnMlaPreprocess(this->atbVariantPack_.workspaceBuffer, this->atbVariantPack_.workspaceBufferSize,
                                      this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn mlaPreprocess execute success.";
    return NO_ERROR;
}
} // namespace atb
