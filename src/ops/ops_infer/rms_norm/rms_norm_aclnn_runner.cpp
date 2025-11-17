/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rms_norm_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/dl_manager.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"
#include "acl/acl.h"
namespace atb {
static const uint32_t IN_TENSOR_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 2;
const uint32_t TENSOR_IDX_ZERO = 0;
const uint32_t TENSOR_IDX_ONE = 1;
static const uint64_t DATASIZE_16BIT = 2;
static const uint64_t DATASIZE_32BIT = 4;

aclnnStatus (*RmsNormAclnnRunner::aclnnGetWorkspaceSizeFunc_)(
        const aclTensor *,// x
        const aclTensor *,// gamma
        double,//epsilon
        const aclTensor *,// yOut
        const aclTensor *,//rstdOut
        uint64_t *,//workspaceSize
        aclOpExecutor ** //executor
    ) = nullptr;

aclnnStatus (*RmsNormAclnnRunner::aclnnExecuteFunc_)(
        void *,//workspace
        uint64_t,//workspaceSize
        aclOpExecutor *,//executor
        aclrtStream//stream
    ) = nullptr;

RmsNormAclnnRunner::RmsNormAclnnRunner(const infer::RmsNormParam &param)
    : AclnnRunner("RmsNormAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormAclnnRunner::RmsNormAclnnRunner called";
}

RmsNormAclnnRunner::~RmsNormAclnnRunner() {}

Status RmsNormAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "RmsNormAclnnRunner LoadMethod";
    if (RmsNormAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        RmsNormAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    static DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
    Status ret = dlManager.getSymbol("aclnnRmsNormGetWorkspaceSize",
                                     (void *&)RmsNormAclnnRunner::aclnnGetWorkspaceSizeFunc_);
    if (ret != NO_ERROR ||
        RmsNormAclnnRunner::aclnnGetWorkspaceSizeFunc_ == nullptr) {
        ATB_LOG(ERROR) << "load aclnnRmsNormGetWorkspaceSize failed! Consider upgrade the CANN first!";
        return ret;
    }
    ret = dlManager.getSymbol("aclnnRmsNorm", (void *&)RmsNormAclnnRunner::aclnnExecuteFunc_);
    if (ret != NO_ERROR ||
        RmsNormAclnnRunner::aclnnExecuteFunc_ == nullptr) {
        ATB_LOG(ERROR) << "load aclnnRmsNorm failed! Consider upgrade the CANN first!";
        return ret;
    }
    ATB_LOG(INFO) << "load aclnnRmsNorm two-staged method success!";
    return NO_ERROR;
}

Status RmsNormAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "RmsNormAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        if (i == IN_TENSOR_NUM - 1) {
            size_t negLen = 0;
            bool notOne = false;
            for (size_t j = 0; j < atbTensor.desc.shape.dimNum; ++j) {
                if (atbTensor.desc.shape.dims[j] != 1 || notOne) {
                    notOne = true;
                    atbTensor.desc.shape.dims[j - negLen] = atbTensor.desc.shape.dims[j];
                } else if (!notOne) {
                    ++negLen;
                }
            }
            atbTensor.desc.shape.dimNum -= negLen;
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
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors.at(i) = aclnnTensorPtr;
    }
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < OUT_TENSOR_NUM - 1; ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "RmsNormAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(TENSOR_IDX_ZERO);
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
    /*rstd build*/
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor atbTensor = {};
    if (param_.normParam.rstd) {
        atbTensor = runnerVariantPack.outTensors.at(TENSOR_IDX_ONE);
    } else {
        atbTensor.desc = runnerVariantPack.inTensors.at(TENSOR_IDX_ZERO).desc;
        atbTensor.desc.dtype = ACL_FLOAT;
        atbTensor.dataSize = runnerVariantPack.inTensors.at(TENSOR_IDX_ZERO).dataSize;
        if (runnerVariantPack.inTensors.at(TENSOR_IDX_ZERO).desc.dtype == ACL_FLOAT) {
            atbTensor.dataSize /= DATASIZE_32BIT;
        } else {
            atbTensor.dataSize /= DATASIZE_16BIT;
        }
        atbTensor.dataSize *= DATASIZE_32BIT;
        for (size_t i = 0; i < atbTensor.desc.shape.dimNum; ++i) {
            if (i >= runnerVariantPack.inTensors.at(TENSOR_IDX_ZERO).desc.shape.dimNum - runnerVariantPack.inTensors.at(TENSOR_IDX_ONE).desc.shape.dimNum) {
                atbTensor.dataSize /= atbTensor.desc.shape.dims[i];
                atbTensor.desc.shape.dims[i] = 1;
            } else {
                atbTensor.desc.shape.dims[i] = runnerVariantPack.inTensors.at(TENSOR_IDX_ZERO).desc.shape.dims[i];
            }
        }
        auto memRes = aclrtMalloc(&atbTensor.deviceData, atbTensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (memRes != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create rstd aclTensor by aclrtMalloc failed!";
            return ERROR_INTERNAL_ERROR;
        }
    }
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    aclnnTensorPtr->tensorIdx = static_cast<int>(TENSOR_IDX_ONE);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclOutTensors.at(TENSOR_IDX_ONE) = aclnnTensorPtr;
    return atb::NO_ERROR;
}

Status RmsNormAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    Status status = RmsNormAclnnRunner::LoadMethod();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return status;
    }
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = RmsNormAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

aclnnStatus RmsNormAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn RmsNorm setup start.";
    Status status = RmsNormAclnnRunner::LoadMethod();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }

    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    aclTensor *x = aclnnVariantPack_.aclInTensors.at(0)->tensor;
    aclTensor *gamma = aclnnVariantPack_.aclInTensors.at(1)->tensor;
    aclTensor *yOut = aclnnVariantPack_.aclOutTensors.at(0)->tensor;
    aclTensor *rstd = aclnnVariantPack_.aclOutTensors.at(1)->tensor;
    double epsilon = (double)param_.normParam.epsilon;
    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << rawExecutorPtr
                  << ", then take the address of the raw ptr: " << &rawExecutorPtr;
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);
    aclnnStatus ret = RmsNormAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        x,
        gamma,
        epsilon,
        yOut,
        rstd,
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



REG_RUNNER_TYPE(RmsNormAclnnRunner);
} // namespace atb