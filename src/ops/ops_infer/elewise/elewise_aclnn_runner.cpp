/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "elewise_aclnn_runner.h"
#include <unordered_map>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM_1 = 1;
static const uint32_t IN_TENSOR_NUM_2 = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
// 初始化类函数指针

// 对应aclnnop/aclnn_cast.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnCastGetWorkspaceSizeFunc_)(const aclTensor *, const aclDataType,
                                                                  const aclTensor *, uint64_t *,
                                                                  aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnCastExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;
// 对应aclnnop/aclnn_cos.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnCosGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                                 aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnCosExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;
// 对应aclnnop/aclnn_sin.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnSinGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                                 aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnSinExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;
// 对应aclnnop/aclnn_neg.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnNegGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                                 aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnNegExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;
// 对应aclnnop/aclnn_logical_not.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnLogicalNotGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *,
                                                                        uint64_t *, aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnLogicalNotExecuteFunc_)(void *, uint64_t, aclOpExecutor *,
                                                               aclrtStream) = nullptr;
// 对应aclnnop/aclnn_logical_and.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnLogicalAndGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *,
                                                                        const aclTensor *, uint64_t *,
                                                                        aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnLogicalAndExecuteFunc_)(void *, uint64_t, aclOpExecutor *,
                                                               aclrtStream) = nullptr;
// 对应aclnnop/aclnn_logical_or.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnLogicalOrGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *,
                                                                       const aclTensor *, uint64_t *,
                                                                       aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnLogicalOrExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;
// 对应aclnnop/aclnn_sub.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnSubGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *,
                                                                 const aclScalar *, const aclTensor *, uint64_t *,
                                                                 aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnSubExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;
// 对应aclnnop/aclnn_eq_tensor.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnEqTensorGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *,
                                                                      const aclTensor *, uint64_t *,
                                                                      aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnEqTensorExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;
// 对应aclnnop/aclnn_tanh.h中的两段式接口
aclnnStatus (*ElewiseAclnnRunner::aclnnTanhGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, uint64_t *,
                                                                  aclOpExecutor **) = nullptr;
aclnnStatus (*ElewiseAclnnRunner::aclnnTanhExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

ElewiseAclnnRunner::ElewiseAclnnRunner(const infer::ElewiseParam &param)
    : AclnnRunner("ElewiseAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ElewiseAclnnRunner::ElewiseAclnnRunner called";
    std::unordered_map<infer::ElewiseParam::ElewiseType, uint32_t> inTensorNumMap = {
        {infer::ElewiseParam::ElewiseType::ELEWISE_CAST, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_COS, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SIN, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_NEG, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_TANH, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SUB, IN_TENSOR_NUM_2},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND, IN_TENSOR_NUM_2},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR, IN_TENSOR_NUM_2},
        {infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL, IN_TENSOR_NUM_2},
    };
    std::unordered_map<infer::ElewiseParam::ElewiseType, uint32_t>::iterator it =
        inTensorNumMap.find(param.elewiseType);
    if (it != inTensorNumMap.end()) {
        inTensorNum_ = it->second;
    }
}

ElewiseAclnnRunner::~ElewiseAclnnRunner() {}

Status ElewiseAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(this->inTensorNum_);
    this->aclnnVariantPack_.aclInTensors.resize(this->inTensorNum_);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
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
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
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

aclnnStatus ElewiseAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn setup start.";
    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorStart = 0;
    aclTensor *self = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *other = (this->inTensorNum_ == IN_TENSOR_NUM_1) ?
                           nullptr :
                           this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    size_t outTensorStart = 0;
    aclTensor *out = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    if (this->aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_FLOAT ||
        this->aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_FLOAT16 ||
        this->aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_BF16) {
        alpha_ = aclCreateScalar(&varAttrFloat_, aclDataType::ACL_FLOAT);
    } else {
        alpha_ = aclCreateScalar(&varAttrInt, aclDataType::ACL_INT32);
    };
    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    aclnnStatus ret = 0;
    switch (param_.elewiseType) {
        break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_CAST:
            ret = ElewiseAclnnRunner::aclnnCastGetWorkspaceSizeFunc_(
                self, param_.outTensorType, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_COS:
            ret = ElewiseAclnnRunner::aclnnCosGetWorkspaceSizeFunc_(
                self, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SIN:
            ret = ElewiseAclnnRunner::aclnnSinGetWorkspaceSizeFunc_(
                self, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_NEG:
            ret = ElewiseAclnnRunner::aclnnNegGetWorkspaceSizeFunc_(
                self, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT:
            ret = ElewiseAclnnRunner::aclnnLogicalNotGetWorkspaceSizeFunc_(
                self, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_TANH:
            ret = ElewiseAclnnRunner::aclnnTanhGetWorkspaceSizeFunc_(
                self, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SUB:
            ret = ElewiseAclnnRunner::aclnnSubGetWorkspaceSizeFunc_(
                self, other, alpha_, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND:
            ret = ElewiseAclnnRunner::aclnnLogicalAndGetWorkspaceSizeFunc_(
                self, other, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR:
            ret = ElewiseAclnnRunner::aclnnLogicalOrGetWorkspaceSizeFunc_(
                self, other, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL:
            ret = ElewiseAclnnRunner::aclnnEqTensorGetWorkspaceSizeFunc_(
                self, other, out, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
            break;
        default:
            ret = 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
            break;
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc failed with return value: " << ret;
        return ret;
    }
    ret = aclDestroyScalar(alpha_);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclDestroyScalar failed with return value: " << ret;
        return ret;
    }
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status ElewiseAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclnnStatus ret = 0;
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    switch (param_.elewiseType) {
        case infer::ElewiseParam::ElewiseType::ELEWISE_CAST:
            if (!ElewiseAclnnRunner::aclnnCastExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnCastExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnCastExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                            this->atbVariantPack_.workspaceBufferSize,
                                                            this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_COS:
            if (!ElewiseAclnnRunner::aclnnCosExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnCosExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnCosExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                           this->atbVariantPack_.workspaceBufferSize,
                                                           this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SIN:
            if (!ElewiseAclnnRunner::aclnnSinExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnSinExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnSinExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                           this->atbVariantPack_.workspaceBufferSize,
                                                           this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_NEG:
            if (!ElewiseAclnnRunner::aclnnNegExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnNegExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnNegExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                           this->atbVariantPack_.workspaceBufferSize,
                                                           this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT:
            if (!ElewiseAclnnRunner::aclnnLogicalNotExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnLogicalNotExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnLogicalNotExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_TANH:
            if (!ElewiseAclnnRunner::aclnnTanhExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnTanhExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnTanhExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                            this->atbVariantPack_.workspaceBufferSize,
                                                            this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SUB:
            if (!ElewiseAclnnRunner::aclnnSubExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnSubExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnSubExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                           this->atbVariantPack_.workspaceBufferSize,
                                                           this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND:
            if (!ElewiseAclnnRunner::aclnnLogicalAndExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnLogicalAndExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnLogicalAndExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR:
            if (!ElewiseAclnnRunner::aclnnLogicalOrExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnLogicalOrExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnLogicalOrExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                 this->atbVariantPack_.workspaceBufferSize,
                                                                 this->aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL:
            if (!ElewiseAclnnRunner::aclnnEqTensorExecuteFunc_) {
                ATB_LOG(ERROR) << GetLogPrefix() << "aclnnEqTensorExecuteFunc_ is null!";
                return ERROR_INVALID_PARAM;
            }
            ret = ElewiseAclnnRunner::aclnnEqTensorExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                this->atbVariantPack_.workspaceBufferSize,
                                                                this->aclnnExecutor_.get(), executeStream);
            break;
        default:
            ret = 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
            break;
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status ElewiseAclnnRunner::LoadMethod(infer::ElewiseParam::ElewiseType elewiseType)
{
    ATB_LOG(INFO) << "ElewiseAclnnRunner LoadMethod";
    Status status = NO_ERROR;
    switch (elewiseType) {
        case infer::ElewiseParam::ElewiseType::ELEWISE_CAST:
            if (ElewiseAclnnRunner::aclnnCastGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnCastExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnCastGetWorkspaceSize", "aclnnCast",
                                                  ElewiseAclnnRunner::aclnnCastGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnCastExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_COS:
            if (ElewiseAclnnRunner::aclnnCosGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnCosExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnCosGetWorkspaceSize", "aclnnCos",
                                                  ElewiseAclnnRunner::aclnnCosGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnCosExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SIN:
            if (ElewiseAclnnRunner::aclnnSinGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnSinExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnSinGetWorkspaceSize", "aclnnSin",
                                                  ElewiseAclnnRunner::aclnnSinGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnSinExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_NEG:
            if (ElewiseAclnnRunner::aclnnNegGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnNegExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnNegGetWorkspaceSize", "aclnnNeg",
                                                  ElewiseAclnnRunner::aclnnNegGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnNegExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT:
            if (ElewiseAclnnRunner::aclnnLogicalNotGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnLogicalNotExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnLogicalNotGetWorkspaceSize", "aclnnLogicalNot",
                                                  ElewiseAclnnRunner::aclnnLogicalNotGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnLogicalNotExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_TANH:
            if (ElewiseAclnnRunner::aclnnTanhGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnTanhExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnTanhGetWorkspaceSize", "aclnnTanh",
                                                  ElewiseAclnnRunner::aclnnTanhGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnTanhExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SUB:
            if (ElewiseAclnnRunner::aclnnSubGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnSubExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnSubGetWorkspaceSize", "aclnnSub",
                                                  ElewiseAclnnRunner::aclnnSubGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnSubExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND:
            if (ElewiseAclnnRunner::aclnnLogicalAndGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnLogicalAndExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnLogicalAndGetWorkspaceSize", "aclnnLogicalAnd",
                                                  ElewiseAclnnRunner::aclnnLogicalAndGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnLogicalAndExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR:
            if (ElewiseAclnnRunner::aclnnLogicalOrGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnLogicalOrExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnLogicalOrGetWorkspaceSize", "aclnnLogicalOr",
                                                  ElewiseAclnnRunner::aclnnLogicalOrGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnLogicalOrExecuteFunc_);
            }
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL:
            if (ElewiseAclnnRunner::aclnnEqTensorGetWorkspaceSizeFunc_ == nullptr ||
                ElewiseAclnnRunner::aclnnEqTensorExecuteFunc_ == nullptr) {
                status = LoadFromSharedObjectFile("aclnnEqTensorGetWorkspaceSize", "aclnnEqTensor",
                                                  ElewiseAclnnRunner::aclnnEqTensorGetWorkspaceSizeFunc_,
                                                  ElewiseAclnnRunner::aclnnEqTensorExecuteFunc_);
            }
            break;
        default:
            break;
    }
    return status;
}

REG_RUNNER_TYPE(ElewiseAclnnRunner);
} // namespace atb
