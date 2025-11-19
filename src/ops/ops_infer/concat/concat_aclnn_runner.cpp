/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "concat_aclnn_runner.h"

#include <aclnn/opdev/op_errno.h>

#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_MINI_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;

}

namespace atb {

// 初始化类函数指针
aclnnStatus (*ConcatAclnnRunner::aclnnGetWorkspaceSizeFunc_)(
    const aclTensorList *inputs,
    int64_t dim,
    aclTensor* output,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) = nullptr;

aclnnStatus (*ConcatAclnnRunner::aclnnExecuteFunc_)(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream) = nullptr;

ConcatAclnnRunner::ConcatAclnnRunner(const infer::ConcatParam &param)
    : AclnnRunner("ConcatAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ConcatAclnnRunner::ConcatAclnnRunner called";
}

Status ConcatAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;

    size_t num_inputs = runnerVariantPack.inTensors.size();
    if (num_inputs < IN_TENSOR_MINI_NUM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Concat requires at least 2 inputs, but got " << num_inputs;
        return ErrorType::ERROR_INVALID_PARAM;
    }

    // 创建输入ACL tensor
    std::vector<aclTensor*> inputTensors;
    inputTensors.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "ConcatAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
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
        inputTensors.push_back(aclnnTensorPtr->tensor);
    }

    aclTensorList* inputTensorList = aclCreateTensorList(inputTensors.data(), inputTensors.size());
    if (inputTensorList == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Failed to create aclTensorList";
        return ACL_ERROR_FAILURE;
    }
    this->aclnnVariantPack_.aclInTensorList.push_back(inputTensorList);

    // 构建输出tensor
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "ConcatAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
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

aclnnStatus ConcatAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn concat setup start.";
    if (!ConcatAclnnRunner::aclnnGetWorkspaceSizeFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    // 调用aclnn获取workspace大小
    aclnnStatus ret = ConcatAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        this->aclnnVariantPack_.aclInTensorList.at(0),           // 输入tensor列表
        param_.concatDim,                                         // concat维度
        this->aclnnVariantPack_.aclOutTensors.at(0)->tensor,      // 输出tensor
        &(this->atbVariantPack_.workspaceBufferSize),             // 输出的workspace大小
        &raw_executor_ptr);                                       // 输出的executor

    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status ConcatAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    if (!ConcatAclnnRunner::aclnnExecuteFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ERROR_INVALID_PARAM;
    }

    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = ConcatAclnnRunner::aclnnExecuteFunc_(
        this->atbVariantPack_.workspaceBuffer,
        this->atbVariantPack_.workspaceBufferSize,
        this->aclnnExecutor_.get(),
        executeStream);

    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }

    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status ConcatAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "ConcatAclnnRunner LoadMethod";
    if (ConcatAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        ConcatAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }

    Status status = LoadFromSharedObjectFile("aclnnCatGetWorkspaceSize", "aclnnCat",
                                             ConcatAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                             ConcatAclnnRunner::aclnnExecuteFunc_);
    return status;
}

REG_RUNNER_TYPE(ConcatAclnnRunner);
} // namespace atb