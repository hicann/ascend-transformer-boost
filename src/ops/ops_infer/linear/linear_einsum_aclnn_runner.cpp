/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_einsum_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include <aclnn/opdev/common_types.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/utils_internal.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const int MATMUL_SELF_ACLNN_TENSOR_IDX = 0;
static const int MATMUL_MAT2_ACLNN_TENSOR_IDX = 1;
static const int MATMUL_OUT_ACLNN_TENSOR_IDX = 0;
static const int DEFAULT_ALIGN = 16;
} // namespace

namespace atb {
AclnnTransposeBatchMatMulGetWorkspaceSizeFunc LinearEinsumAclnnRunner::aclnnTransposeBatchMatMulGetWorkspaceSizeFunc_ =
    nullptr;
AclnnTransposeBatchMatMulExecuteFunc LinearEinsumAclnnRunner::aclnnTransposeBatchMatMulExecuteFunc_ = nullptr;
AclnnTransposeBatchMatMulGetWorkspaceSizeFunc
    LinearEinsumAclnnRunner::aclnnTransposeBatchMatMulWeightNzGetWorkspaceSizeFunc_ = nullptr;
AclnnTransposeBatchMatMulExecuteFunc LinearEinsumAclnnRunner::aclnnTransposeBatchMatMulWeightNzExecuteFunc_ = nullptr;

LinearEinsumAclnnRunner::LinearEinsumAclnnRunner(const infer::LinearParam &param)
    : AclnnRunner("LinearEinsumAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearEinsumAclnnRunner::LinearEinsumAclnnRunner";

    GetTensorNum();
}

LinearEinsumAclnnRunner::~LinearEinsumAclnnRunner()
{
    DestroyPermArrays();
}

Status LinearEinsumAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "LinearEinsumAclnnRunner::LoadAclnnFuncs";
    if ((aclnnTransposeBatchMatMulGetWorkspaceSizeFunc_ && aclnnTransposeBatchMatMulExecuteFunc_) &&
        (aclnnTransposeBatchMatMulWeightNzGetWorkspaceSizeFunc_ && aclnnTransposeBatchMatMulWeightNzExecuteFunc_)) {
        return NO_ERROR;
    }
    Status st = NO_ERROR;
    if (!aclnnTransposeBatchMatMulGetWorkspaceSizeFunc_ || !aclnnTransposeBatchMatMulExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnTransposeBatchMatMulGetWorkspaceSize", "aclnnTransposeBatchMatMul",
                                      aclnnTransposeBatchMatMulGetWorkspaceSizeFunc_,
                                      aclnnTransposeBatchMatMulExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (!aclnnTransposeBatchMatMulWeightNzGetWorkspaceSizeFunc_ || !aclnnTransposeBatchMatMulWeightNzExecuteFunc_) {
        st = LoadFromSharedObjectFile(
            "aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize", "aclnnTransposeBatchMatMulWeightNz",
            aclnnTransposeBatchMatMulWeightNzGetWorkspaceSizeFunc_, aclnnTransposeBatchMatMulWeightNzExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

Status LinearEinsumAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearEinsumAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: "
                  << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    isWeightNz_ = runnerVariantPack.inTensors[1].desc.format == ACL_FORMAT_FRACTAL_NZ;
    InitTensorIndex();
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclOutTensors.reserve(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(aclOutTensorNum_);
    Status st = CreateMatmulSelfAclnnTensor();
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
    return NO_ERROR;
}

aclnnStatus LinearEinsumAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearEinsumAclnnRunner::SetAclNNWorkspaceExecutor";
    DestroyPermArrays();
    Status st = CreatePermArrays();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreatePermArrays failed";
        return ACL_ERROR_FAILURE;
    }
    if (isWeightNz_) {
        return SetAclnnTransposeBatchMatMulWeightNzWorkspaceExecutor();
    }
    return SetAclnnTransposeBatchMatMulWorkspaceExecutor();
}

Status LinearEinsumAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearEinsumAclnnRunner::LaunchAclnnKernel";

    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);

    aclnnStatus ret = ACL_SUCCESS;
    if (isWeightNz_) {
        ret = aclnnTransposeBatchMatMulWeightNzExecuteFunc_(
            atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    } else {
        ret = aclnnTransposeBatchMatMulExecuteFunc_(
            atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

void LinearEinsumAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = 2; // self, mat2
    aclOutTensorNum_ = 1;
}

void LinearEinsumAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorIndex_ = 0;
    matmulSelfAclTensorIndex_ = 0;
    matmulMat2AclTensorIndex_ = 0;
    matmulOutAclTensorIndex_ = 0;
}

Status LinearEinsumAclnnRunner::CreateMatmulSelfAclnnTensor()
{
    ATB_LOG(INFO) << "LinearEinsumAclnnRunner::CreateMatmulSelfAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateXAclnnTensor(MATMUL_SELF_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul self aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    matmulSelfAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearEinsumAclnnRunner::CreateMatmulMat2AclnnTensor()
{
    ATB_LOG(INFO) << "LinearEinsumAclnnRunner::CreateMatmulMat2AclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = isWeightNz_ ?
                                                      CreateWeightNzAclnnTensor(MATMUL_MAT2_ACLNN_TENSOR_IDX) :
                                                      CreateWeightAclnnTensor(MATMUL_MAT2_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul mat2 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    matmulMat2AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearEinsumAclnnRunner::CreateMatmulOutAclnnTensor()
{
    ATB_LOG(INFO) << "LinearEinsumAclnnRunner::CreateMatmulOutAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateOutputAclnnTensor(MATMUL_OUT_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul out aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    matmulOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

Status LinearEinsumAclnnRunner::CreatePermArrays()
{
    int64_t permX1Vals[3] = {1, 0, 2};
    int64_t permX2Vals[3];
    if (param_.transposeB) {
        permX2Vals[0] = 0; permX2Vals[1] = 2; permX2Vals[2] = 1;
    } else {
        permX2Vals[0] = 0; permX2Vals[1] = 1; permX2Vals[2] = 2;
    }
    int64_t permYVals[3] = {1, 0, 2};

    permX1_ = aclCreateIntArray(permX1Vals, 3);
    permX2_ = aclCreateIntArray(permX2Vals, 3);
    permY_ = aclCreateIntArray(permYVals, 3);

    if (!permX1_ || !permX2_ || !permY_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclCreateIntArray failed";
        DestroyPermArrays();
        return ERROR_INTERNAL_ERROR;
    }
    return NO_ERROR;
}

void LinearEinsumAclnnRunner::DestroyPermArrays()
{
    if (permX1_) {
        aclDestroyIntArray(permX1_);
        permX1_ = nullptr;
    }
    if (permX2_) {
        aclDestroyIntArray(permX2_);
        permX2_ = nullptr;
    }
    if (permY_) {
        aclDestroyIntArray(permY_);
        permY_ = nullptr;
    }
}

aclnnStatus LinearEinsumAclnnRunner::SetAclnnTransposeBatchMatMulWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearEinsumAclnnRunner::SetAclnnTransposeBatchMatMulWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(matmulSelfAclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(matmulMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(matmulOutAclTensorIndex_)->tensor;

    int8_t cubeMathType = 0;
    int32_t batchSplitFactor = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnTransposeBatchMatMulGetWorkspaceSizeFunc_(
        self, mat2, nullptr, nullptr, permX1_, permX2_, permY_, cubeMathType, batchSplitFactor, out,
        &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

aclnnStatus LinearEinsumAclnnRunner::SetAclnnTransposeBatchMatMulWeightNzWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearEinsumAclnnRunner::SetAclnnTransposeBatchMatMulWeightNzWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(matmulSelfAclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(matmulMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(matmulOutAclTensorIndex_)->tensor;

    int8_t cubeMathType = 0;
    int32_t batchSplitFactor = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnTransposeBatchMatMulWeightNzGetWorkspaceSizeFunc_(
        self, mat2, nullptr, nullptr, permX1_, permX2_, permY_, cubeMathType, batchSplitFactor, out,
        &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

std::shared_ptr<AclNNTensor> LinearEinsumAclnnRunner::CreateXAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearEinsumAclnnRunner::CreateWeightAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearEinsumAclnnRunner::CreateWeightNzAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    Dims storageShape = atbTensor.desc.shape;
    {
        Dims oldShape = viewShape;
        viewShape.dims[0] = oldShape.dims[0];
        viewShape.dims[1] = oldShape.dims[2];
        viewShape.dims[2] = oldShape.dims[1] * oldShape.dims[3];
        viewShape.dimNum = 3;
        ATB_LOG(INFO) << GetLogPrefix() << "viewShape: [" << viewShape.dims[0] << ", " << viewShape.dims[1] << ", "
                      << viewShape.dims[2] << "]";
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    {
        Dims oldShape = storageShape;
        storageShape.dims[0] = oldShape.dims[0];
        storageShape.dims[1] = 1;
        storageShape.dims[2] = oldShape.dims[1];
        storageShape.dims[3] = oldShape.dims[2];
        storageShape.dims[4] = oldShape.dims[3];
        storageShape.dimNum = 5;
        ATB_LOG(INFO) << GetLogPrefix() << "storageShape: [" << storageShape.dims[0] << ", " << storageShape.dims[1]
                      << ", " << storageShape.dims[2] << ", " << storageShape.dims[3] << ", " << storageShape.dims[4]
                      << "]";
    }
    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearEinsumAclnnRunner::CreateOutputAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearEinsumAclnnRunner::InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex)
{
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->tensorIdx = aclnnTensorIndex;
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    return aclnnTensorPtr;
}

REG_RUNNER_TYPE(LinearEinsumAclnnRunner);
} // namespace atb
