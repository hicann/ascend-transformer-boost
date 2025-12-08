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
} // namespace

namespace atb {
aclnnStatus (*SortAclnnRunner::aclnnGetWorkspaceSizeFunc_)(const aclTensor *, int64_t, int64_t, bool, bool, aclTensor *,
                                                           aclTensor *, uint64_t *, aclOpExecutor **) = nullptr;
aclnnStatus (*SortAclnnRunner::aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

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
    return atb::NO_ERROR;
}

aclnnStatus SortAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Topk setup start.";
    if (SortAclnnRunner::aclnnGetWorkspaceSizeFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Topk, aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorStart = 0;
    aclTensor *x = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor; // self
    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;  // valueOut
    aclTensor *indices = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor; // indicesOut
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
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status SortAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "SortAclnnRunner LoadMethod";
    if (SortAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr && SortAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnTopkGetWorkspaceSize", "aclnnTopk",
                                    SortAclnnRunner::aclnnGetWorkspaceSizeFunc_, SortAclnnRunner::aclnnExecuteFunc_);
}

REG_RUNNER_TYPE(SortAclnnRunner);
} // namespace atb
