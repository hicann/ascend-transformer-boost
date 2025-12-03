/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const int MATMUL_SELF_ACLNN_TENSOR_IDX = 0;
static const int MATMUL_MAT2_ACLNN_TENSOR_IDX = 1;
static const int MATMUL_OUT_ACLNN_TENSOR_IDX = 0;
static const int ADDMM_SELF_ACLNN_TENSOR_IDX = 0;
static const int ADDMM_MAT1_ACLNN_TENSOR_IDX = 1;
static const int ADDMM_MAT2_ACLNN_TENSOR_IDX = 2;
static const int ADDMM_OUT_ACLNN_TENSOR_IDX = 0;
}

namespace atb {
AclnnMatmulGetWorkspaceSizeFunc LinearAclnnRunner::aclnnMatmulGetWorkspaceSizeFunc_ = nullptr;
AclnnMatmulExecuteFunc LinearAclnnRunner::aclnnMatmulExecuteFunc_ = nullptr;
AclnnAddmmGetWorkspaceSizeFunc LinearAclnnRunner::aclnnAddmmGetWorkspaceSizeFunc_ = nullptr;
AclnnAddmmExecuteFunc LinearAclnnRunner::aclnnAddmmExecuteFunc_ = nullptr;

LinearAclnnRunner::LinearAclnnRunner(const infer::LinearParam &param) : AclnnRunner("LinearAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::LinearAclnnRunner";

    GetTensorNum();
    InitTensorIndex();
}

LinearAclnnRunner::~LinearAclnnRunner()
{
    if (alpha_) {
        if (aclDestroyScalar(alpha_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "alpha aclDestroyScalar error";
        }
        alpha_ = nullptr;
    }
    if (beta_) {
        if (aclDestroyScalar(beta_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "beta aclDestroyScalar error";
        }
        beta_ = nullptr;
    }
}

Status LinearAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::LoadAclnnFuncs";
    if (aclnnMatmulGetWorkspaceSizeFunc_ && aclnnMatmulExecuteFunc_ && aclnnAddmmGetWorkspaceSizeFunc_ &&
        aclnnAddmmExecuteFunc_) {
        return NO_ERROR;
    }
    Status st = NO_ERROR;
    if (!aclnnMatmulGetWorkspaceSizeFunc_ || !aclnnMatmulExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnMatmulGetWorkspaceSize", "aclnnMatmul", aclnnMatmulGetWorkspaceSizeFunc_,
            aclnnMatmulExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (!aclnnAddmmGetWorkspaceSizeFunc_ || !aclnnAddmmExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnAddmmGetWorkspaceSize", "aclnnAddmm", aclnnAddmmGetWorkspaceSizeFunc_,
            aclnnAddmmExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

Status LinearAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: "
                  << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclOutTensors.reserve(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(aclOutTensorNum_);
    Status st = NO_ERROR;
    if (param_.hasBias) {
        st = CreateAddmmMat1AclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateAddmmMat2AclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateAddmmSelfAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateAddmmOutAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
    } else {
        st = CreateMatmulSelfAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateMatmulMat2AclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateMatmulOutAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

aclnnStatus LinearAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::SetAclNNWorkspaceExecutor";

    if (param_.hasBias) {
        return SetAclnnAddmmWorkspaceExecutor();
    } else {
        return SetAclnnMatmulWorkspaceExecutor();
    }
}

Status LinearAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::LaunchAclnnKernel";

    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);

    aclnnStatus ret = ACL_SUCCESS;
    if (param_.hasBias) {
        ret = aclnnAddmmExecuteFunc_(
            atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    } else {
        ret = aclnnMatmulExecuteFunc_(
            atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

void LinearAclnnRunner::GetTensorNum()
{
    if (param_.hasBias) {
        aclInTensorNum_ = 3; // 3: self, mat1, mat2
    } else {
        aclInTensorNum_ = 2;  // 2: self, mat2
    }
    aclOutTensorNum_ = 1;
}

void LinearAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorIndex_ = 0;
    matmulSelfAclTensorIndex_ = 0;
    matmulMat2AclTensorIndex_ = 0;
    matmulOutAclTensorIndex_ = 0;
    addmmSelfAclTensorIndex_ = 0;
    addmmMat1AclTensorIndex_ = 0;
    addmmMat2AclTensorIndex_ = 0;
    addmmOutAclTensorIndex_ = 0;
}

Status LinearAclnnRunner::CreateMatmulSelfAclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateMatmulSelfAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateXAclnnTensor(MATMUL_SELF_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul self aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    matmulSelfAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateMatmulMat2AclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateMatmulMat2AclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateWeightAclnnTensor(MATMUL_MAT2_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul mat2 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    matmulMat2AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateMatmulOutAclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateMatmulOutAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateOutputAclnnTensor(MATMUL_OUT_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul out aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    matmulOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateAddmmSelfAclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateAddmmSelfAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateBiasAclnnTensor(ADDMM_SELF_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "addmm self aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    addmmSelfAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateAddmmMat1AclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateAddmmMat1AclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateXAclnnTensor(ADDMM_MAT1_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "addmm mat1 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    addmmMat2AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateAddmmMat2AclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateAddmmMat2AclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateWeightAclnnTensor(ADDMM_MAT2_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "addmm mat2 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    addmmMat2AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateAddmmOutAclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateAddmmOutAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateOutputAclnnTensor(ADDMM_OUT_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "addmm out aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    addmmOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

aclnnStatus LinearAclnnRunner::SetAclnnMatmulWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::SetAclnnMatmulWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(matmulSelfAclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(matmulMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(matmulOutAclTensorIndex_)->tensor;

    int8_t cubeMathType = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();
    
    aclnnStatus ret = aclnnMatmulGetWorkspaceSizeFunc_(self, mat2, out, cubeMathType,
        &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

aclnnStatus LinearAclnnRunner::SetAclnnAddmmWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::SetAclnnAddmmWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(addmmSelfAclTensorIndex_)->tensor;
    aclTensor *mat1 = aclnnVariantPack_.aclInTensors.at(addmmMat1AclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(addmmMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(addmmOutAclTensorIndex_)->tensor;

    if (alpha_) {
        if (aclDestroyScalar(alpha_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "alpha aclDestroyScalar error";
            return ERROR_INTERNAL_ERROR;
        }
        alpha_ = nullptr;
    }
    if (beta_) {
        if (aclDestroyScalar(beta_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "beta aclDestroyScalar error";
            return ERROR_INTERNAL_ERROR;
        }
        beta_ = nullptr;
    }

    float alphaValue = 1.0f;
    alpha_ = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    float betaValue = 1.0f;
    beta_ = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
    int8_t cubeMathType = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();
    
    aclnnStatus ret = aclnnAddmmGetWorkspaceSizeFunc_(self, mat1, mat2, beta_, alpha_, out, cubeMathType,
        &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateXAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    if (viewShape.dimNum == 3 && atbVariantPack_.inTensors.at(1).desc.shape.dimNum == 2) {
        viewShape.dims[0] = viewShape.dims[0] * viewShape.dims[1];
        viewShape.dims[1] = viewShape.dims[2];
        viewShape.dimNum = 2;
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    if (param_.transposeA) {
        int64_t m = viewShape.dims[viewShape.dimNum - 1];
        int64_t k = viewShape.dims[viewShape.dimNum - 2];
        viewShape.dims[viewShape.dimNum - 2] = m;
        viewShape.dims[viewShape.dimNum - 1] = k;
        int64_t lastStride = aclnnTensorPtr->strides[viewShape.dimNum - 1];
        aclnnTensorPtr->strides[viewShape.dimNum - 1] = aclnnTensorPtr->strides[viewShape.dimNum - 2];
        aclnnTensorPtr->strides[viewShape.dimNum - 2] = lastStride;
    }
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateWeightAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    if (param_.transposeB) {
        int64_t k = viewShape.dims[viewShape.dimNum - 1];
        int64_t n = viewShape.dims[viewShape.dimNum - 2];
        viewShape.dims[viewShape.dimNum - 2] = k;
        viewShape.dims[viewShape.dimNum - 1] = n;
        int64_t lastStride = aclnnTensorPtr->strides[viewShape.dimNum - 1];
        aclnnTensorPtr->strides[viewShape.dimNum - 1] = aclnnTensorPtr->strides[viewShape.dimNum - 2];
        aclnnTensorPtr->strides[viewShape.dimNum - 2] = lastStride;
    }
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateBiasAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateOutputAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    if (atbVariantPack_.inTensors.at(0).desc.shape.dimNum == 3 &&
        atbVariantPack_.inTensors.at(1).desc.shape.dimNum == 2) {
        viewShape.dims[0] = viewShape.dims[0] * viewShape.dims[1];
        viewShape.dims[1] = viewShape.dims[2];
        viewShape.dimNum = 2;
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex)
{
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->tensorIdx = aclnnTensorIndex;
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    return aclnnTensorPtr;
}

REG_RUNNER_TYPE(LinearAclnnRunner);
}  // namespace atb
