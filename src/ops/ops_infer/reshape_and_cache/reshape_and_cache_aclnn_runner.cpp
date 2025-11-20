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
#include "atb/utils/dl_manager.h"
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

aclnnStatus (*ReshapeAndCacheAclnnRunner::aclnnGetWorkspaceSizeFunc_)(
        const aclTensor *,   // key
        const aclTensor *,   // keyCacheRef
        const aclTensor *,   // slotMapping
        const aclTensor *,   // value
        const aclTensor *,   // valueCacheRef
        const aclTensor *,   // compressLensOptional
        const aclTensor *,   // compressSeqOffsetOptional
        const aclTensor *,   // seqLensOptional
        char *,              // cacheModeOptional
        char *,              // scatterModeOptional
        const aclIntArray *, // stridesOptional
        const aclIntArray *, // offsetsOptional
        uint64_t *,          // workspaceSize
        aclOpExecutor **     // executor
    ) = nullptr;

aclnnStatus (*ReshapeAndCacheAclnnRunner::aclnnExecuteFunc_)(
        void *,//workspace
        uint64_t,//workspaceSize
        aclOpExecutor *,//executor
        aclrtStream//stream
    ) = nullptr;

ReshapeAndCacheAclnnRunner::ReshapeAndCacheAclnnRunner(const infer::ReshapeAndCacheParam &param)
    : AclnnRunner("ReshapeAndCacheAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheAclnnRunner::ReshapeAndCacheAclnnRunner called";
}

ReshapeAndCacheAclnnRunner::~ReshapeAndCacheAclnnRunner() {}

Status ReshapeAndCacheAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "ReshapeAndCacheAclnnRunner LoadMethod";
    if (ReshapeAndCacheAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        ReshapeAndCacheAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    static DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
    Status ret = dlManager.getSymbol("aclnnScatterPaKvCacheGetWorkspaceSize",
                                     (void *&)ReshapeAndCacheAclnnRunner::aclnnGetWorkspaceSizeFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnScatterPaKvCacheGetWorkspaceSize failed! Consider upgrade the CANN first!";
        return ret;
    }
    ret = dlManager.getSymbol("aclnnScatterPaKvCache", (void *&)ReshapeAndCacheAclnnRunner::aclnnExecuteFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnScatterPaKvCache failed! Consider upgrade the CANN first!";
        return ret;
    }
    ATB_LOG(INFO) << "load aclnnScatterPaKvCache two-staged method success!";
    return NO_ERROR;
}

Status ReshapeAndCacheAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
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
        this->aclnnVariantPack_.aclInTensors.at(i) = aclnnTensorPtr;
    }
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
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
        this->aclnnVariantPack_.aclOutTensors.at(i) = aclnnTensorPtr;
    }
    return atb::NO_ERROR;
}

aclnnStatus ReshapeAndCacheAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn ReshapeAndCache setup start.";
    Status status = ReshapeAndCacheAclnnRunner::LoadMethod();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    }
    size_t inTensorStart = 0;
    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    this->aclnnVariantPack_.aclInTensors.at(1)->tensorIdx = 3;
    this->aclnnVariantPack_.aclInTensors.at(2)->tensorIdx = 1;
    this->aclnnVariantPack_.aclInTensors.at(3)->tensorIdx = 4;
    this->aclnnVariantPack_.aclInTensors.at(4)->tensorIdx = 2;
    aclTensor *key = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *value = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *keyCacheRef = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *valueCacheRef = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *slotMapping = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *compressLensOptional = nullptr;
    aclTensor *compressSeqOffsetOptional = nullptr;
    aclTensor *seqLensOptional = nullptr;
    const aclIntArray *stridesOptional = nullptr;
    const aclIntArray *offsetsOptional = nullptr;
    std::string cacheMode = "Norm";
    std::string scatterMode = "None";
    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << rawExecutorPtr
                  << ", then take the address of the raw ptr: " << &rawExecutorPtr;

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    aclnnStatus ret = ReshapeAndCacheAclnnRunner::aclnnGetWorkspaceSizeFunc_(
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
        &(this->atbVariantPack_.workspaceBufferSize),
        &rawExecutorPtr);
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status ReshapeAndCacheAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    Status status = ReshapeAndCacheAclnnRunner::LoadMethod();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return status;
    }
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = ReshapeAndCacheAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

REG_RUNNER_TYPE(ReshapeAndCacheAclnnRunner);
} // namespace atb