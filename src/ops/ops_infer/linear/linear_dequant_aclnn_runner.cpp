/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_dequant_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/utils_internal.h"
#include "atb/utils/operation_register.h"

static constexpr int64_t DEFAULT_ALIGN = 16;
static constexpr int64_t INT8_ALIGN = 32;
static constexpr int MATMUL_SELF_ACLNN_TENSOR_IDX = 0;
static constexpr int MATMUL_MAT2_ACLNN_TENSOR_IDX = 1;
static constexpr int MATMUL_OUT_ACLNN_TENSOR_IDX = 0;

namespace atb {
AclnnQuantMatmulV5GetWorkspaceSizeFunc LinearDequantAclnnRunner::aclnnQuantMatmulV5GetWorkspaceSizeFunc_ = nullptr;
AclnnQuantMatmulV5ExecuteFunc LinearDequantAclnnRunner::aclnnQuantMatmulV5ExecuteFunc_ = nullptr;
AclnnQuantMatmulWeightNzGetWorkspaceSizeFunc LinearDequantAclnnRunner::aclnnQuantMatmulWeightNzGetWorkspaceSizeFunc_ =
    nullptr;
AclnnQuantMatmulWeightNzExecuteFunc LinearDequantAclnnRunner::aclnnQuantMatmulWeightNzExecuteFunc_ = nullptr;

LinearDequantAclnnRunner::LinearDequantAclnnRunner(const infer::LinearParam &param)
    : AclnnRunner("LinearDequantAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearDequantAclnnRunner::LinearDequantAclnnRunner";
    GetTensorNum();
}

LinearDequantAclnnRunner::~LinearDequantAclnnRunner() {}

Status LinearDequantAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::LoadAclnnFuncs";
    if (aclnnQuantMatmulV5GetWorkspaceSizeFunc_ && aclnnQuantMatmulV5ExecuteFunc_ &&
        aclnnQuantMatmulWeightNzGetWorkspaceSizeFunc_ && aclnnQuantMatmulWeightNzExecuteFunc_) {
        return NO_ERROR;
    }
    Status st = NO_ERROR;
    if (!aclnnQuantMatmulV5GetWorkspaceSizeFunc_ || !aclnnQuantMatmulV5ExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnQuantMatmulV5GetWorkspaceSize", "aclnnQuantMatmulV5",
                                      aclnnQuantMatmulV5GetWorkspaceSizeFunc_, aclnnQuantMatmulV5ExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (!aclnnQuantMatmulWeightNzGetWorkspaceSizeFunc_ || !aclnnQuantMatmulWeightNzExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnQuantMatmulWeightNzGetWorkspaceSize", "aclnnQuantMatmulWeightNz",
                                      aclnnQuantMatmulWeightNzGetWorkspaceSizeFunc_,
                                      aclnnQuantMatmulWeightNzExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

Status LinearDequantAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearDequantAclnnRunner::BuildAclnnVariantPack"
                  << ", inTensors.size=" << runnerVariantPack.inTensors.size();
    atbVariantPack_ = runnerVariantPack;
    isWeightNz_ = runnerVariantPack.inTensors[1].desc.format == ACL_FORMAT_FRACTAL_NZ;

    InitTensorIndex();

    size_t actualInTensorNum = runnerVariantPack.inTensors.size();
    if (actualInTensorNum > aclInTensorNum_) {
        aclInTensorNum_ = actualInTensorNum;
    }

    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclOutTensors.reserve(1);
    aclnnVariantPack_.aclOutTensors.resize(1);

    Status st = CreateXAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateWeightAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.hasBias) {
        st = CreateBiasAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (param_.quantMode == infer::LinearParam::PER_TOKEN) {
        st = CreateDeqScaleAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreatePerTokenScaleAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
    } else {
        st = CreateDeqScaleAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
    }
    st = CreateOutAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

aclnnStatus LinearDequantAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearDequantAclnnRunner::SetAclNNWorkspaceExecutor";
    if (isWeightNz_) {
        return SetAclnnQuantMatmulWeightNzWorkspaceExecutor();
    }
    return SetAclnnQuantMatmulWorkspaceExecutor();
}

Status LinearDequantAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearDequantAclnnRunner::LaunchAclnnKernel";

    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);

    aclnnStatus ret = ACL_SUCCESS;
    if (isWeightNz_) {
        ret = aclnnQuantMatmulWeightNzExecuteFunc_(atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize,
                                                   aclnnExecutor_.get(), executeStream);
    } else {
        ret = aclnnQuantMatmulV5ExecuteFunc_(atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize,
                                             aclnnExecutor_.get(), executeStream);
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn quant matmul kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

void LinearDequantAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = param_.hasBias ? 4 : 3;
    if (param_.quantMode == infer::LinearParam::PER_TOKEN) {
        aclInTensorNum_ = param_.hasBias ? 5 : 4;
    }
}

void LinearDequantAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    xAclTensorIndex_ = 0;
    weightAclTensorIndex_ = 0;
    descaleAclTensorIndex_ = 0;
    biasAclTensorIndex_ = 0;
    perTokenScaleAclTensorIndex_ = 0;
    outAclTensorIndex_ = 0;
}

Status LinearDequantAclnnRunner::CreateXAclnnTensor()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::CreateXAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, MATMUL_SELF_ACLNN_TENSOR_IDX);
    Dims viewShape = atbTensor.desc.shape;
    Dims storageShape = atbTensor.desc.shape;

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

    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreateXAclnnTensor: aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    xAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearDequantAclnnRunner::CreateWeightAclnnTensor()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::CreateWeightAclnnTensor, isWeightNz=" << isWeightNz_;

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, MATMUL_MAT2_ACLNN_TENSOR_IDX);
    Dims viewShape = atbTensor.desc.shape;
    Dims storageShape = atbTensor.desc.shape;

    if (isWeightNz_) {
        if (viewShape.dimNum == 2) {
            Dims oldShape = viewShape;
            storageShape.dims[0] = 1;
            storageShape.dims[1] = UtilsInternal::AlignUp(oldShape.dims[1], INT8_ALIGN) / INT8_ALIGN;
            storageShape.dims[2] = UtilsInternal::AlignUp(oldShape.dims[0], DEFAULT_ALIGN);
            storageShape.dims[3] = INT8_ALIGN;
            storageShape.dimNum = 4;
            ATB_LOG(INFO) << GetLogPrefix() << "Nz storageShape: [1, " << storageShape.dims[1] << ", "
                          << storageShape.dims[2] << ", " << storageShape.dims[3] << "]";
        } else if (viewShape.dimNum == 3) {
            Dims oldShape = viewShape;
            storageShape.dims[0] = oldShape.dims[0];
            storageShape.dims[1] = 1;
            storageShape.dims[2] = UtilsInternal::AlignUp(oldShape.dims[2], INT8_ALIGN) / INT8_ALIGN;
            storageShape.dims[3] = UtilsInternal::AlignUp(oldShape.dims[1], DEFAULT_ALIGN);
            storageShape.dims[4] = INT8_ALIGN;
            storageShape.dimNum = 5;
            ATB_LOG(INFO) << GetLogPrefix() << "Nz storageShape batch";
        }
        if (viewShape.dimNum == 4) {
            Dims oldShape = viewShape;
            viewShape.dims[0] = oldShape.dims[2];
            viewShape.dims[1] = oldShape.dims[1] * oldShape.dims[3];
            viewShape.dimNum = 2;
        } else if (viewShape.dimNum == 3) {
            viewShape.dims[0] = viewShape.dims[0];
            viewShape.dims[1] = viewShape.dims[2];
            viewShape.dimNum = 2;
        }
    }

    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    if (!param_.transposeB) {
        int64_t k = viewShape.dims[viewShape.dimNum - 1];
        int64_t n = viewShape.dims[viewShape.dimNum - 2];
        viewShape.dims[viewShape.dimNum - 2] = k;
        viewShape.dims[viewShape.dimNum - 1] = n;
        int64_t lastStride = aclnnTensorPtr->strides[viewShape.dimNum - 1];
        aclnnTensorPtr->strides[viewShape.dimNum - 1] = aclnnTensorPtr->strides[viewShape.dimNum - 2];
        aclnnTensorPtr->strides[viewShape.dimNum - 2] = lastStride;
    }

    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreateWeightAclnnTensor: aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    weightAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearDequantAclnnRunner::CreateDeqScaleAclnnTensor()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::CreateDeqScaleAclnnTensor";

    size_t totalInTensors = atbVariantPack_.inTensors.size();
    if (!param_.hasBias && param_.quantMode != infer::LinearParam::PER_TOKEN &&
        totalInTensors > static_cast<size_t>(param_.hasBias ? 4 : 3)) {
        atbInTensorIndex_ = totalInTensors - 1;
        ATB_LOG(INFO) << GetLogPrefix() << "multi-tensor dequant, use last tensor as deqScale, index="
                      << atbInTensorIndex_;
    }

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, MATMUL_MAT2_ACLNN_TENSOR_IDX + 1);
    Dims viewShape = atbTensor.desc.shape;

    if (viewShape.dimNum == 2 && viewShape.dims[0] == 1) {
        viewShape.dims[0] = viewShape.dims[1];
        viewShape.dimNum = 1;
    }

    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreateDeqScaleAclnnTensor: aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    descaleAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearDequantAclnnRunner::CreateBiasAclnnTensor()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::CreateBiasAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, MATMUL_MAT2_ACLNN_TENSOR_IDX + 2);
    Dims viewShape = atbTensor.desc.shape;

    if (viewShape.dimNum == 2 && viewShape.dims[0] == 1) {
        viewShape.dims[0] = viewShape.dims[1];
        viewShape.dimNum = 1;
    }

    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreateBiasAclnnTensor: aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    biasAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearDequantAclnnRunner::CreatePerTokenScaleAclnnTensor()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::CreatePerTokenScaleAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, MATMUL_MAT2_ACLNN_TENSOR_IDX + 3);
    Dims viewShape = atbTensor.desc.shape;

    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreatePerTokenScaleAclnnTensor: aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    perTokenScaleAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearDequantAclnnRunner::CreateOutAclnnTensor()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::CreateOutAclnnTensor";

    Tensor atbTensor = atbVariantPack_.outTensors.at(0);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, MATMUL_OUT_ACLNN_TENSOR_IDX);
    Dims viewShape = atbTensor.desc.shape;

    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreateOutAclnnTensor: aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(0) = aclnnTensorPtr;
    outAclTensorIndex_ = 0;
    return NO_ERROR;
}

aclnnStatus LinearDequantAclnnRunner::SetAclnnQuantMatmulWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::SetAclnnQuantMatmulWorkspaceExecutor";

    aclTensor *x1 = aclnnVariantPack_.aclInTensors.at(xAclTensorIndex_)->tensor;
    aclTensor *x2 = aclnnVariantPack_.aclInTensors.at(weightAclTensorIndex_)->tensor;
    aclTensor *x1Scale = nullptr;
    aclTensor *x2Scale = aclnnVariantPack_.aclInTensors.at(descaleAclTensorIndex_)->tensor;
    aclTensor *bias = nullptr;

    if (param_.quantMode == infer::LinearParam::PER_TOKEN) {
        x1Scale = aclnnVariantPack_.aclInTensors.at(perTokenScaleAclTensorIndex_)->tensor;
    }
    if (param_.hasBias) {
        bias = aclnnVariantPack_.aclInTensors.at(biasAclTensorIndex_)->tensor;
    }

    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(0)->tensor;

    bool transposeX1 = false;
    bool transposeX2 = true;
    int64_t groupSize = 0;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnQuantMatmulV5GetWorkspaceSizeFunc_(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr,
                                                              nullptr, bias, transposeX1, transposeX2, groupSize, out,
                                                              &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

aclnnStatus LinearDequantAclnnRunner::SetAclnnQuantMatmulWeightNzWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearDequantAclnnRunner::SetAclnnQuantMatmulWeightNzWorkspaceExecutor";

    aclTensor *x1 = aclnnVariantPack_.aclInTensors.at(xAclTensorIndex_)->tensor;
    aclTensor *x2 = aclnnVariantPack_.aclInTensors.at(weightAclTensorIndex_)->tensor;
    aclTensor *x1Scale = nullptr;
    aclTensor *x2Scale = aclnnVariantPack_.aclInTensors.at(descaleAclTensorIndex_)->tensor;
    aclTensor *bias = nullptr;

    if (param_.quantMode == infer::LinearParam::PER_TOKEN) {
        x1Scale = aclnnVariantPack_.aclInTensors.at(perTokenScaleAclTensorIndex_)->tensor;
    }
    if (param_.hasBias) {
        bias = aclnnVariantPack_.aclInTensors.at(biasAclTensorIndex_)->tensor;
    }

    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(0)->tensor;

    bool transposeX1 = false;
    bool transposeX2 = true;
    int64_t groupSize = 0;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnQuantMatmulWeightNzGetWorkspaceSizeFunc_(
        x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, bias, transposeX1, transposeX2, groupSize, out,
        &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

std::shared_ptr<AclNNTensor> LinearDequantAclnnRunner::InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex)
{
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->tensorIdx = aclnnTensorIndex;
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    return aclnnTensorPtr;
}

REG_RUNNER_TYPE(LinearDequantAclnnRunner);
} // namespace atb
