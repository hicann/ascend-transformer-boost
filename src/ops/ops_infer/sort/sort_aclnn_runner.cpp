/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"
#include "sort_aclnn_runner.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t INDEX_ONE = 1;
} // namespace

namespace atb {
AclnnGetWorkspaceSizeFunc SortAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
AclnnExecuteFunc SortAclnnRunner::aclnnExecuteFunc_ = nullptr;
AclnnCastGetWorkspaceSizeFunc SortAclnnRunner::aclnnCastGetWorkspaceSizeFunc_ = nullptr;
AclnnCastExecuteFunc SortAclnnRunner::aclnnCastExecuteFunc_ = nullptr;

SortAclnnRunner::SortAclnnRunner(const infer::SortParam &param) : AclnnRunner("SortAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SortAclnnRunner::SortAclnnRunner created";
}

SortAclnnRunner::~SortAclnnRunner() {}

Status SortAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // self
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
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
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    // output
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
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
        this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }

    indices_ = std::make_shared<AclNNTensor>();
    atb::Tensor atbTensor = runnerVariantPack.outTensors.at(INDEX_ONE);
    indices_->atbTensor = atbTensor;
    indices_->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, indices_,
                                ACL_INT64);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create int64 indices by aclCreateTensor failed!";
        return ret;
    }
    return atb::NO_ERROR;
}

aclnnStatus SortAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Topk setup start.";

    size_t inTensorStart = 0;
    aclTensor *x = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor; // self
    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;  // valueOut
    aclTensor *indices = indices_->tensor; // indicesOut
    
    int64_t k = static_cast<int64_t>(param_.num.at(0));
    int64_t dim = -1;
    bool largest = true;
    bool sorted = true;

    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    aclnnStatus ret = SortAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        x, k, dim, largest, sorted, output, indices, &(this->atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret != ACL_SUCCESS) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;

    aclOpExecutor *rawCastExecutorPtr = this->aclnnCastExecutor_.get();
    aclTensor *out = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor; // indicesOut
    ret = SortAclnnRunner::aclnnCastGetWorkspaceSizeFunc_(
        indices_->tensor, ACL_INT32, out, &(this->castworkspacesize_), &rawCastExecutorPtr);
    this->aclnnCastExecutor_ = std::shared_ptr<aclOpExecutor>(rawCastExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret != ACL_SUCCESS) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "aclnnCastGetWorkspaceSize failed!";
        return ret;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->castworkspacesize_;
    return ret;
}

Status SortAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";

    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = SortAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                         this->atbVariantPack_.workspaceBufferSize,
                                                         this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }

    aclrtStream castExecuteStream = GetExecuteStream(this->atbVariantPack_.context);
    ret = SortAclnnRunner::aclnnCastExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                         this->castworkspacesize_,
                                                         this->aclnnCastExecutor_.get(), castExecuteStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status SortAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "SortAclnnRunner LoadMethod";
    if (SortAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr && SortAclnnRunner::aclnnExecuteFunc_ != nullptr
    && SortAclnnRunner::aclnnCastGetWorkspaceSizeFunc_ != nullptr && SortAclnnRunner::aclnnCastExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status ret = LoadFromSharedObjectFile("aclnnTopkGetWorkspaceSize", "aclnnTopk",
                                    SortAclnnRunner::aclnnGetWorkspaceSizeFunc_, SortAclnnRunner::aclnnExecuteFunc_);
    if (ret != NO_ERROR) {
        return ret;
    }

    return LoadFromSharedObjectFile("aclnnCastGetWorkspaceSize", "aclnnCast",
                                    SortAclnnRunner::aclnnCastGetWorkspaceSizeFunc_, SortAclnnRunner::aclnnCastExecuteFunc_);
}

REG_RUNNER_TYPE(SortAclnnRunner);
} // namespace atb
