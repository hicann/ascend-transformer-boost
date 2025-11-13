/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_add_rms_norm_runner.h"

#include <aclnn/opdev/op_errno.h>
#include "atb/utils/dl_manager.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "acl/acl.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 3;
static const uint32_t OUT_TENSOR_NUM = 2;

} // namespace

namespace atb {
// 初始化类函数指针
aclnnStatus (*AclnnAddRmsNormRunner::aclnnGetWorkspaceSizeFunc_)(
    const aclTensor *, const aclTensor *, const aclTensor *, double,
    const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **) = nullptr;

aclnnStatus (*AclnnAddRmsNormRunner::aclnnExecuteFunc_)( void *, uint64_t , aclOpExecutor *, const aclrtStream) = nullptr;

AclnnAddRmsNormRunner::AclnnAddRmsNormRunner(const infer::RmsNormParam &param)
    : AclnnRunner("AclnnAddRmsNormRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AclnnAddRmsNormRunner::AclnnAddRmsNormRunner called";
}

AclnnAddRmsNormRunner::~AclnnAddRmsNormRunner()
{
    aclrtFree(rstdTensor.deviceData);
}

Status AclnnAddRmsNormRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    RunnerVariantPack tmp_pack(runnerVariantPack);
    size_t actual_gamma_dim = 0;
    size_t gamma_ones = 0;
    for (size_t i = 0; i < runnerVariantPack.inTensors.at(2).desc.shape.dimNum; i++) {
        if (runnerVariantPack.inTensors.at(2).desc.shape.dims[i] == 1) {
            gamma_ones += 1;
        } else {
            actual_gamma_dim += 1;
        }
    }

    // if rstd == False, we only normalize on the last dimension;
    if (!param_.normParam.rstd) {
        if (actual_gamma_dim > 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "rstd is False, we only normalize on the last dimension."
                << "But the input gamma has dimension of "
                << actual_gamma_dim << ".";
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
    }

    // reshape the tensor dim
    tmp_pack.inTensors.at(2).desc.shape.dimNum = actual_gamma_dim;
    for (size_t i = 0; i < actual_gamma_dim; i++) {
        tmp_pack.inTensors.at(2).desc.shape.dims[i] = runnerVariantPack.inTensors.at(2).desc.shape.dims[i + gamma_ones];
    }

    this->atbVariantPack_ = tmp_pack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "AclnnAddRmsNormRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = tmp_pack.inTensors.at(i);
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

    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "AclnnAddRmsNormRunner::BuildAclnnVariantPack outTensor index: " << i;
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

    // create a local rstdTensor, will not return
    rstdTensor = this->atbVariantPack_.inTensors.at(0);
    rstdTensor.desc.dtype = ACL_FLOAT;
    
    size_t xDimNum = this->atbVariantPack_.inTensors.at(0).desc.shape.dimNum;
    size_t gammaDimNum = this->atbVariantPack_.inTensors.at(2).desc.shape.dimNum;
    for (size_t i = 0; i < gammaDimNum; i++) {
        // set the normalized dimension to 1
        rstdTensor.desc.shape.dims[xDimNum - gammaDimNum + i] = 1;
    }

    uint64_t resDataSize = sizeof(float);
    for (size_t i = 0; i < rstdTensor.desc.shape.dimNum; i++) {
        // set the normalized dimension to 1
        resDataSize *= rstdTensor.desc.shape.dims[i];
    }
    rstdTensor.dataSize = resDataSize;
    rstdTensor.deviceData = nullptr;
    auto aclrtRet = aclrtMalloc(&rstdTensor.deviceData, rstdTensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclrtRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << "AclnnAddRmsNorm outTensor malloc fail";
        return atb::ERROR_INTERNAL_ERROR;
    }

    rstdAclnnTensor = std::make_shared<AclNNTensor>();
    rstdAclnnTensor->atbTensor = rstdTensor;
    rstdAclnnTensor->strides = GetCopyTensorStride(rstdTensor.desc.shape);
    ret = CallAclCreateTensor(rstdTensor.desc.shape, rstdTensor.desc.shape, rstdTensor, rstdAclnnTensor,
                                rstdTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    rstdAclnnTensor->tensorIdx = static_cast<int>(2);
    rstdAclnnTensor->needUpdateTensorDataPtr = false;

    return atb::NO_ERROR;
}

aclnnStatus AclnnAddRmsNormRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnnAddRmsNorm setup start.";
    Status loadStatus =  AclnnAddRmsNormRunner::LoadMethod();
    if (loadStatus != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    if (!aclnnGetWorkspaceSizeFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }

    ATB_LOG(INFO) << GetLogPrefix() << "aclnn addRmsNorm: "
                  << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << raw_executor_ptr
                  << ", then take the address of the raw ptr: " << &raw_executor_ptr;

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    aclnnStatus ret = AclnnAddRmsNormRunner::aclnnGetWorkspaceSizeFunc_(
        this->aclnnVariantPack_.aclInTensors.at(0)->tensor,    // x1
        this->aclnnVariantPack_.aclInTensors.at(1)->tensor,    // x2
        this->aclnnVariantPack_.aclInTensors.at(2)->tensor,    // gamma
        (double)param_.preNormParam.epsilon,
        this->aclnnVariantPack_.aclOutTensors.at(0)->tensor,    // yOut
        rstdAclnnTensor->tensor,                                // rstdOut
        this->aclnnVariantPack_.aclOutTensors.at(1)->tensor,    // xOut
        &(this->atbVariantPack_.workspaceBufferSize),
        &raw_executor_ptr);
    
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status AclnnAddRmsNormRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";

    Status loadStatus =  AclnnAddRmsNormRunner::LoadMethod();
    if (loadStatus != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    if (!aclnnExecuteFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ERROR_INVALID_PARAM;
    }
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = AclnnAddRmsNormRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status AclnnAddRmsNormRunner::LoadMethod()
{
    ATB_LOG(INFO) << "AclnnAddRmsNormRunner LoadMethod";
    if (AclnnAddRmsNormRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        AclnnAddRmsNormRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    static DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
    Status ret = dlManager.getSymbol("aclnnAddRmsNormGetWorkspaceSize",
                                     (void *&)AclnnAddRmsNormRunner::aclnnGetWorkspaceSizeFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load AclnnAddRmsNormGetWorkspaceSize failed! Consider upgrade the CANN first!";
        return ret;
    }
    ret = dlManager.getSymbol("aclnnAddRmsNorm", (void *&)AclnnAddRmsNormRunner::aclnnExecuteFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnAddRmsNorm failed! Consider upgrade the CANN first!";
        return ret;
    }
    ATB_LOG(INFO) << "load AclnnAddRmsNorm two-staged method success!";
    return NO_ERROR;
}

REG_RUNNER_TYPE(AclnnAddRmsNormRunner);
} // namespace atb
