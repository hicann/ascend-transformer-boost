/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "split_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
} // namespace

namespace atb {

SplitTensorFuncType SplitAclnnRunner::splitGetWorkspaceSizeFunc_ = nullptr;
ExecuteFuncType SplitAclnnRunner::splitExecuteFunc_ = nullptr;
SplitWithSizeFuncType SplitAclnnRunner::splitWithSizeGetWorkspaceSizeFunc_ = nullptr;
ExecuteFuncType SplitAclnnRunner::splitWithSizeExecuteFunc_ = nullptr;

SplitAclnnRunner::SplitAclnnRunner(const infer::SplitParam &param) : AclnnRunner("SplitAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SplitAclnnRunner::SplitAclnnRunner created";
    AssignFunc();
}

SplitAclnnRunner::~SplitAclnnRunner() {}

Status SplitAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // x
    aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < aclnnVariantPack_.aclInTensors.size(); ++i) {
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
        aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    // output1, output2, output3(optional)
    int32_t outTensorNum = param_.splitNum;
    aclnnVariantPack_.aclOutTensors.reserve(outTensorNum);
    aclnnVariantPack_.aclOutTensors.resize(outTensorNum);
    std::vector<aclTensor *> outTensors = {};
    for (size_t i = 0; i < aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack outTensor index: " << i;
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
        // use tensorList
        aclnnTensorPtr->tensorListidx = static_cast<int>(0);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
        outTensors.emplace_back(aclnnTensorPtr->tensor);
    }
    aclTensorList *outTensorList = aclCreateTensorList(outTensors.data(), outTensors.size());
    aclnnVariantPack_.aclOutTensorList.push_back(outTensorList);
    return atb::NO_ERROR;
}

void SplitAclnnRunner::AssignFunc()
{
    if (param_.splitSizes.size() != 0) {
        executeFunc_ = SplitAclnnRunner::splitExecuteFunc_;
        splitWithSize_ = true;
    } else {
        executeFunc_ = SplitAclnnRunner::splitWithSizeExecuteFunc_;
    }
}

aclnnStatus SplitAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Split setup start.";
    AssignFunc();
    aclTensor *x = aclnnVariantPack_.aclInTensors.at(0)->tensor; // self
    aclTensorList *outTensorList = aclnnVariantPack_.aclOutTensorList.at(0); // outTensorList
    int32_t splitNum = param_.splitNum;
    Dims xDims = aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.shape;
    int64_t dim = param_.splitDim;
    if (dim < 0) {
        dim += xDims.dimNum;
    }
    uint64_t splitSections = static_cast<uint64_t>(xDims.dims[dim] / splitNum);
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    aclnnStatus ret = ACL_SUCCESS;
    if (splitWithSize_) {
        size_t size = param_.splitSizes.size();
        int64_t splitSizeData[size];
        for (size_t i = 0; i < size; ++i) {
            splitSizeData[i] = param_.splitSizes[i];
        }
        if (splitSizeArray_) {
            ret = aclDestroyIntArray(splitSizeArray_);
            if (ret != ACL_SUCCESS) {
                return ret;
            }
            splitSizeArray_ = nullptr;
        }
        splitSizeArray_ = aclCreateIntArray(splitSizeData, size);
        ret = splitWithSizeGetWorkspaceSizeFunc_(x, splitSizeArray_, dim, outTensorList,
                                                       &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    } else {
        ret = splitGetWorkspaceSizeFunc_(x, splitSections, dim, outTensorList,
                                               &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    }
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status SplitAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    AssignFunc();
    void *executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret =
        executeFunc_(atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize,
                           aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    if (splitSizeArray_) {
        if (aclDestroyIntArray(splitSizeArray_) != ACL_SUCCESS) {
            return ERROR_CANN_ERROR;
        }
        splitSizeArray_ = nullptr;
    }
    return NO_ERROR;
}

Status SplitAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "SplitAclnnRunner LoadAclnnFuncs";
    Status status = LoadFromSharedObjectFile("aclnnSplitWithSizeGetWorkspaceSize", "aclnnSplitWithSize",
                                             SplitAclnnRunner::splitWithSizeGetWorkspaceSizeFunc_,
                                             SplitAclnnRunner::splitWithSizeExecuteFunc_);
    if (status != NO_ERROR) {
        return status;
    }
    return LoadFromSharedObjectFile("aclnnSplitTensorGetWorkspaceSize", "aclnnSplitTensor",
                                    SplitAclnnRunner::splitGetWorkspaceSizeFunc_, SplitAclnnRunner::splitExecuteFunc_);
}

REG_RUNNER_TYPE(SplitAclnnRunner);
} // namespace atb
