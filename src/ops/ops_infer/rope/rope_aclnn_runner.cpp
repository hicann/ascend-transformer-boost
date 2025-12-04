/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rope_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"
#include "acl/acl.h"
namespace atb {
static const uint32_t ROPE_IN_NUM = 4;
static const uint32_t ROPE_OUT_NUM = 2;
static const uint32_t ROPE_QUERY_INDEX = 0;
static const uint32_t ROPE_KEY_INDEX = 1;
static const uint32_t ROPE_COS_INDEX = 2;
static const uint32_t ROPE_SIN_INDEX = 3;
static const uint32_t ROTARY_COEFF_HALF = 2;
static const uint32_t ROTARY_COEFF_QUARTER = 4;
static const uint32_t ACLNN_INPUT_DIM = 4;
static const uint32_t DIM_B = 0;
static const uint32_t DIM_S = 1;
static const uint32_t DIM_N = 2;
static const uint32_t DIM_D = 3;
static const uint32_t DIM_ONE = 1;
static const uint32_t COS_SIN_NUM = 2;
static const int64_t LAYOUT_BSND = 1;
aclnnGetWorkspaceSizeFuncPtr RopeAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
aclnnExecuteFuncPtr RopeAclnnRunner::aclnnExecuteFunc_ = nullptr;

RopeAclnnRunner::RopeAclnnRunner(const infer::RopeParam &param)
    : AclnnRunner("RopeAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "RopeAclnnRunner::RopeAclnnRunner called";
}

RopeAclnnRunner::~RopeAclnnRunner() {}

Status RopeAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "RopeAclnnRunner LoadMethod";
    if (RopeAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        RopeAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status status = LoadFromSharedObjectFile("aclnnApplyRotaryPosEmbV2GetWorkspaceSize", "aclnnApplyRotaryPosEmbV2",
                                             RopeAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                             RopeAclnnRunner::aclnnExecuteFunc_);
    return status;
}

Status RopeAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    this->aclnnVariantPack_.aclInTensors.reserve(ROPE_IN_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(ROPE_IN_NUM);
    this->aclnnVariantPack_.aclOutTensors.reserve(ROPE_OUT_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(ROPE_OUT_NUM);
    Status ret = NO_ERROR;
    //key and query reshape
    int64_t dSize = runnerVariantPack.inTensors.at(ROPE_COS_INDEX).desc.shape.dims[DIM_S];
    for (size_t i = 0; i < ROPE_IN_NUM - COS_SIN_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "RopeAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        if (runnerVariantPack.inTensors.at(ROPE_QUERY_INDEX).desc.shape.dimNum != ACLNN_INPUT_DIM) {
            atbTensor.desc.shape.dimNum = ACLNN_INPUT_DIM;
            atbTensor.desc.shape.dims[DIM_N] = atbTensor.desc.shape.dims[DIM_S] / dSize;
            atbTensor.desc.shape.dims[DIM_D] = dSize;
            atbTensor.desc.shape.dims[DIM_S] = DIM_ONE;
        } else {
            atbTensor.desc.shape.dims[DIM_N] = atbTensor.desc.shape.dims[DIM_N] * atbTensor.desc.shape.dims[DIM_D] / dSize;
            atbTensor.desc.shape.dims[DIM_D] = dSize;
        }
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                      atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = false;
        this->aclnnVariantPack_.aclInTensors.at(i) = aclnnTensorPtr;
    }
    //cos sin reshape
    int64_t bSize = runnerVariantPack.inTensors.at(ROPE_QUERY_INDEX).desc.shape.dims[DIM_B];
    int64_t sSize = runnerVariantPack.inTensors.at(ROPE_QUERY_INDEX).desc.shape.dims[DIM_S];
    for (size_t i = ROPE_IN_NUM - COS_SIN_NUM; i < ROPE_IN_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "RopeAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        atbTensor.desc.shape.dimNum = ACLNN_INPUT_DIM;
        atbTensor.desc.shape.dims[DIM_D] = atbTensor.desc.shape.dims[DIM_S];
        atbTensor.desc.shape.dims[DIM_N] = DIM_ONE;
        if (runnerVariantPack.inTensors.at(ROPE_QUERY_INDEX).desc.shape.dimNum != ACLNN_INPUT_DIM) {
            atbTensor.desc.shape.dims[DIM_S] = DIM_ONE;
        } else {
            atbTensor.desc.shape.dims[DIM_B] = bSize;
            atbTensor.desc.shape.dims[DIM_S] = sSize;
        }
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                      atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = false;
        this->aclnnVariantPack_.aclInTensors.at(i) = aclnnTensorPtr;
    }
    //create output
    for (size_t i = 0; i < ROPE_OUT_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "RopeAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = this->aclnnVariantPack_.aclInTensors.at(i)->atbTensor;
        atbTensor.deviceData = runnerVariantPack.outTensors.at(i).deviceData;
        auto memRet = aclrtMemcpy(atbTensor.deviceData, atbTensor.dataSize,
                    this->aclnnVariantPack_.aclInTensors.at(i)->atbTensor.deviceData, atbTensor.dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (memRet != ACL_SUCCESS) {
            return ERROR_CANN_ERROR;
        }
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                      atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = false;
        this->aclnnVariantPack_.aclOutTensors.at(i) = aclnnTensorPtr;
    }
    return NO_ERROR;
}

Status RopeAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = RopeAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

aclnnStatus RopeAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn rope setup start.";
    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    aclTensor *queryRef = aclnnVariantPack_.aclOutTensors.at(ROPE_QUERY_INDEX)->tensor;
    aclTensor *keyRef = aclnnVariantPack_.aclOutTensors.at(ROPE_KEY_INDEX)->tensor;
    aclTensor *cos = aclnnVariantPack_.aclInTensors.at(ROPE_COS_INDEX)->tensor;
    aclTensor *sin = aclnnVariantPack_.aclInTensors.at(ROPE_SIN_INDEX)->tensor;
    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << rawExecutorPtr
                  << ", then take the address of the raw ptr: " << &rawExecutorPtr;
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);
    std::string rotaryMode = "half";
    if (param_.rotaryCoeff == ROTARY_COEFF_HALF) {
        rotaryMode = "half";
    } else if (param_.rotaryCoeff == ROTARY_COEFF_QUARTER) {
        rotaryMode = "quarter";
    } else {
        rotaryMode = "interleave";
    }
    aclnnStatus ret = RopeAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        queryRef,
        keyRef,
        cos,
        sin,
        LAYOUT_BSND,
        (char *)rotaryMode.c_str(),
        &(this->atbVariantPack_.workspaceBufferSize),
        &rawExecutorPtr
    );
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

REG_RUNNER_TYPE(RopeAclnnRunner);
} // namespace atb