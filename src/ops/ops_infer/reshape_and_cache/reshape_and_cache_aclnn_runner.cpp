/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reshape_and_cache_aclnn_runner.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"
#include "acl/acl.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 5;
static const uint32_t OUT_TENSOR_NUM = 2;
const uint32_t TENSOR_IDX_ZERO = 0;
const uint32_t TENSOR_IDX_ONE = 1;
static const int DIM0 = 0;
static const int DIM1 = 1;

AclnnGetWorkspaceSizeFunc ReshapeAndCacheAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
AclnnExecuteFunc ReshapeAndCacheAclnnRunner::aclnnExecuteFunc_ = nullptr;

ReshapeAndCacheAclnnRunner::ReshapeAndCacheAclnnRunner(const infer::ReshapeAndCacheParam &param)
    : AclnnRunner("ReshapeAndCacheAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheAclnnRunner::ReshapeAndCacheAclnnRunner called";
}

ReshapeAndCacheAclnnRunner::~ReshapeAndCacheAclnnRunner() {}

Status ReshapeAndCacheAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "ReshapeAndCacheAclnnRunner LoadMethod";
    Status status = NO_ERROR;
    if (aclnnGetWorkspaceSizeFunc_ == nullptr ||
        aclnnExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnScatterPaKvCacheGetWorkspaceSize", "aclnnScatterPaKvCache",
                                           aclnnGetWorkspaceSizeFunc_,
                                           aclnnExecuteFunc_);
    }
    return status;
}

Status ReshapeAndCacheAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < IN_TENSOR_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                      atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        aclnnVariantPack_.aclInTensors.at(i) = aclnnTensorPtr;
    }
    aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < OUT_TENSOR_NUM; ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        aclnnVariantPack_.aclOutTensors.at(i) = aclnnTensorPtr;
    }
    return atb::NO_ERROR;
}

aclnnStatus ReshapeAndCacheAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn ReshapeAndCache setup start.";
    size_t inTensorIndex = 0;
    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();
    aclnnVariantPack_.aclInTensors.at(1)->tensorIdx = 3;
    aclnnVariantPack_.aclInTensors.at(2)->tensorIdx = 1;
    aclnnVariantPack_.aclInTensors.at(3)->tensorIdx = 4;
    aclnnVariantPack_.aclInTensors.at(4)->tensorIdx = 2;
    aclTensor *key = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    aclTensor *value = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    aclTensor *keyCacheRef = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    aclTensor *valueCacheRef = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    aclTensor *slotMapping = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    aclTensor *compressLensOptional = nullptr;
    aclTensor *compressSeqOffsetOptional = nullptr;
    aclTensor *seqLensOptional = nullptr;
    const aclIntArray *stridesOptional = nullptr;
    const aclIntArray *offsetsOptional = nullptr;
    std::string cacheMode = "Norm";
    std::string scatterMode = "None";
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(aclnnExecutor_): " << &(aclnnExecutor_)
                  << ", addr of aclnnExecutor_: " << aclnnExecutor_
                  << ", raw ptr from it: " << rawExecutorPtr
                  << ", then take the address of the raw ptr: " << &rawExecutorPtr;

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(atbVariantPack_.workspaceBufferSize);

    aclnnStatus ret = aclnnGetWorkspaceSizeFunc_(
        key,
        keyCacheRef,
        slotMapping,
        value,
        valueCacheRef,
        compressLensOptional,
        compressSeqOffsetOptional,
        seqLensOptional,
        (char *)cacheMode.c_str(),
        (char *)scatterMode.c_str(),
        stridesOptional,
        offsetsOptional,
        &(atbVariantPack_.workspaceBufferSize),
        &rawExecutorPtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status ReshapeAndCacheAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                        atbVariantPack_.workspaceBufferSize,
                                        aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

REG_RUNNER_TYPE(ReshapeAndCacheAclnnRunner);
} // namespace atb