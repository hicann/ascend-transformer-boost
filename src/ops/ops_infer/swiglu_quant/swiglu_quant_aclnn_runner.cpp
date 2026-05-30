/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "swiglu_quant_aclnn_runner.h"

#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"
#include "acl/acl.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 2;
static const char *QUANT_MODE_DYNAMIC = "dynamic";
static const int64_t GROUP_LIST_TYPE = 0;
static const int64_t DST_TYPE_INT8 = 2;
} // namespace

namespace atb {
AclnnSwiGluQuantV2GetWorkspaceSizeFunc SwigluQuantAclnnRunner::aclnnSwiGluQuantV2GetWorkspaceSizeFunc_ = nullptr;
AclnnSwiGluQuantV2Func SwigluQuantAclnnRunner::aclnnSwiGluQuantV2Func_ = nullptr;
AclnnInplaceReciprocalGetWorkspaceSizeFunc SwigluQuantAclnnRunner::aclnnInplaceReciprocalGetWorkspaceSizeFunc_ =
    nullptr;
AclnnInplaceReciprocalFunc SwigluQuantAclnnRunner::aclnnInplaceReciprocalFunc_ = nullptr;

SwigluQuantAclnnRunner::SwigluQuantAclnnRunner(const infer::SwigluQuantParam &param)
    : AclnnRunner("SwigluQuantAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SwigluQuantAclnnRunner::SwigluQuantAclnnRunner called";
}

SwigluQuantAclnnRunner::~SwigluQuantAclnnRunner()
{
    FreeSmoothScales();
}

void SwigluQuantAclnnRunner::FreeSmoothScales()
{
    if (smoothScalesAclnnTensor_ != nullptr && smoothScalesAclnnTensor_->tensor != nullptr) {
        aclDestroyTensor(smoothScalesAclnnTensor_->tensor);
        smoothScalesAclnnTensor_->tensor = nullptr;
        smoothScalesAclnnTensor_ = nullptr;
    }
    if (smoothScalesDeviceAddr_ != nullptr) {
        aclrtFree(smoothScalesDeviceAddr_);
        smoothScalesDeviceAddr_ = nullptr;
    }
}

bool SwigluQuantAclnnRunner::useCache()
{
    return false;
}

Status SwigluQuantAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "SwigluQuantAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
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
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "SwigluQuantAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
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

    FreeSmoothScales();

    int64_t hiddenSize =
        runnerVariantPack.outTensors.at(0).desc.shape.dims[runnerVariantPack.outTensors.at(0).desc.shape.dimNum - 1];
    int64_t smoothScalesSize = hiddenSize * static_cast<int64_t>(sizeof(float));
    std::vector<float> smoothScalesHost(hiddenSize, 1.0f);
    aclError aclRet = aclrtMalloc(&smoothScalesDeviceAddr_, smoothScalesSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "alloc smoothScales device memory failed!";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    aclRet = aclrtMemcpy(smoothScalesDeviceAddr_, smoothScalesSize, smoothScalesHost.data(), smoothScalesSize,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "copy smoothScales to device failed!";
        return ERROR_CANN_ERROR;
    }

    smoothScalesAclnnTensor_ = std::make_shared<AclNNTensor>();
    std::vector<int64_t> smoothScalesShape = {hiddenSize};
    std::vector<int64_t> smoothScalesStrides = {1};
    smoothScalesAclnnTensor_->tensor =
        aclCreateTensor(smoothScalesShape.data(), static_cast<int64_t>(smoothScalesShape.size()), ACL_FLOAT,
                        smoothScalesStrides.data(), 0, ACL_FORMAT_ND, smoothScalesShape.data(),
                        static_cast<int64_t>(smoothScalesShape.size()), smoothScalesDeviceAddr_);
    if (smoothScalesAclnnTensor_->tensor == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create smoothScales aclTensor failed!";
        return ERROR_INTERNAL_ERROR;
    }

    return atb::NO_ERROR;
}

aclnnStatus SwigluQuantAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn swigluQuantV2 setup start.";
    Status loadStatus = SwigluQuantAclnnRunner::LoadMethod();
    if (loadStatus != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    if (!aclnnSwiGluQuantV2GetWorkspaceSizeFunc_ || !aclnnInplaceReciprocalGetWorkspaceSizeFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    ATB_LOG(INFO) << " aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorStart = 0;
    aclTensor *x = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    size_t outTensorStart = 0;
    aclTensor *yOut = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    aclTensor *scaleOut = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;

    aclnnStatus ret = ACL_SUCCESS;
    aclTensor *smoothScales = smoothScalesAclnnTensor_->tensor;

    this->swigluQuantWorkspaceSize_ = 0;
    aclOpExecutor *rawSwigluExecutorPtr = this->aclnnExecutor_.get();
    ret = aclnnSwiGluQuantV2GetWorkspaceSizeFunc_(
        x, smoothScales, nullptr, nullptr, true, const_cast<char *>(QUANT_MODE_DYNAMIC), GROUP_LIST_TYPE, DST_TYPE_INT8,
        yOut, scaleOut, &(this->swigluQuantWorkspaceSize_), &rawSwigluExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnSwiGluQuantV2GetWorkspaceSize failed!";
        return ret;
    }
    ret = aclSetAclOpExecutorRepeatable(rawSwigluExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Set SwiGluQuant AclOpExecutorRepeatable failed!";
        return ret;
    }
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawSwigluExecutorPtr, [](aclOpExecutor *ptr) {
        if (ptr) {
            aclDestroyAclOpExecutor(ptr);
        }
    });

    this->inplaceReciprocalWorkspaceSize_ = 0;
    aclOpExecutor *rawInplaceReciprocalExecutorPtr = this->aclnnInplaceReciprocalExecutor_.get();
    ret = aclnnInplaceReciprocalGetWorkspaceSizeFunc_(scaleOut, &(this->inplaceReciprocalWorkspaceSize_),
                                                      &rawInplaceReciprocalExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnInplaceReciprocalGetWorkspaceSize failed!";
        return ret;
    }
    ret = aclSetAclOpExecutorRepeatable(rawInplaceReciprocalExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Set InplaceReciprocal AclOpExecutorRepeatable failed!";
        return ret;
    }
    this->aclnnInplaceReciprocalExecutor_ =
        std::shared_ptr<aclOpExecutor>(rawInplaceReciprocalExecutorPtr, [](aclOpExecutor *ptr) {
            if (ptr) {
                aclDestroyAclOpExecutor(ptr);
            }
        });

    this->atbVariantPack_.workspaceBufferSize =
        this->swigluQuantWorkspaceSize_ + this->inplaceReciprocalWorkspaceSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "swigluQuantWorkspaceSize_: " << this->swigluQuantWorkspaceSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "inplaceReciprocalWorkspaceSize_: " << this->inplaceReciprocalWorkspaceSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status SwigluQuantAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    Status loadStatus = SwigluQuantAclnnRunner::LoadMethod();
    if (loadStatus != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return ERROR_CANN_ERROR;
    }
    if (!aclnnSwiGluQuantV2Func_ || !aclnnInplaceReciprocalFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ERROR_INVALID_PARAM;
    }
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret =
        aclnnSwiGluQuantV2Func_(this->atbVariantPack_.workspaceBuffer, this->swigluQuantWorkspaceSize_,
                                this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnSwiGluQuantV2 launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ret = aclnnInplaceReciprocalFunc_(this->atbVariantPack_.workspaceBuffer + this->swigluQuantWorkspaceSize_,
                                      this->inplaceReciprocalWorkspaceSize_, this->aclnnInplaceReciprocalExecutor_.get(),
                                      executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnInplaceReciprocal launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status SwigluQuantAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "SwigluQuantAclnnRunner::LoadMethod";
    if (aclnnSwiGluQuantV2GetWorkspaceSizeFunc_ && aclnnSwiGluQuantV2Func_ &&
        aclnnInplaceReciprocalGetWorkspaceSizeFunc_ && aclnnInplaceReciprocalFunc_) {
        return NO_ERROR;
    }
    Status status = LoadFromSharedObjectFile("aclnnSwiGluQuantV2GetWorkspaceSize", "aclnnSwiGluQuantV2",
                                             aclnnSwiGluQuantV2GetWorkspaceSizeFunc_, aclnnSwiGluQuantV2Func_);
    if (status != NO_ERROR) {
        return status;
    }
    return LoadFromSharedObjectFile("aclnnInplaceReciprocalGetWorkspaceSize", "aclnnInplaceReciprocal",
                                    aclnnInplaceReciprocalGetWorkspaceSizeFunc_, aclnnInplaceReciprocalFunc_);
}

REG_RUNNER_TYPE(SwigluQuantAclnnRunner);
} // namespace atb
