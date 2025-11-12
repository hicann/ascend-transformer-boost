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
#include "atb/utils/dl_manager.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace atb {
aclnnStatus (*LinearAclnnRunner::aclnnMatmulGetWorkspaceSizeFunc_)(
    const aclTensor *, const aclTensor *, const aclTensor *, int8_t, uint64_t *, aclOpExecutor **) = nullptr;

aclnnStatus (*LinearAclnnRunner::aclnnMatmulExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

aclnnStatus (*LinearAclnnRunner::aclnnAddmmGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *,
    const aclTensor *, aclScalar *, aclScalar *, const aclTensor *, int8_t, uint64_t *, aclOpExecutor **) = nullptr;

aclnnStatus (*LinearAclnnRunner::aclnnAddmmExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

LinearAclnnRunner::LinearAclnnRunner(const infer::LinearParam &param) : AclnnRunner("LinearAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::LinearAclnnRunner called";
}

LinearAclnnRunner::~LinearAclnnRunner()
{
    if (alpha_) {
        if (aclDestroyScalar(alpha_) != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclDestroyScalar alpha error";
        }
    }
    if (beta_) {
        if (aclDestroyScalar(beta_) != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclDestroyScalar beta error";
        }
    }
}

Status LinearAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::LoadMethod";
    if (aclnnMatmulGetWorkspaceSizeFunc_ && aclnnMatmulExecuteFunc_ && aclnnAddmmGetWorkspaceSizeFunc_ &&
        aclnnAddmmExecuteFunc_) {
        return NO_ERROR;
    }
    static DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
    Status st = dlManager.getSymbol(
        "aclnnMatmulGetWorkspaceSize", (void *&)LinearAclnnRunner::aclnnMatmulGetWorkspaceSizeFunc_);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnMatmulGetWorkspaceSize failed! Consider upgrade the CANN first!";
        return st;
    }
    st = dlManager.getSymbol("aclnnMatmul", (void *&)LinearAclnnRunner::aclnnMatmulExecuteFunc_);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnMatmul failed! Consider upgrade the CANN first!";
        return st;
    }
    st = dlManager.getSymbol("aclnnAddmmGetWorkspaceSize", (void *&)LinearAclnnRunner::aclnnAddmmGetWorkspaceSizeFunc_);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnAddmmGetWorkspaceSize failed! Consider upgrade the CANN first!";
        return st;
    }
    st = dlManager.getSymbol("aclnnAddmm", (void *&)LinearAclnnRunner::aclnnAddmmExecuteFunc_);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnAddmm failed! Consider upgrade the CANN first!";
        return st;
    }
    ATB_LOG(INFO) << "load two-staged method success!";
    return NO_ERROR;
}

Status LinearAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::BuildAclnnVariantPack"
                  << "runnerVariantPack: " << runnerVariantPack.ToString();
    GetInTensorNum();
    atbVariantPack_ = runnerVariantPack;
    aclnnVariantPack_.aclInTensors.reserve(inTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(inTensorNum_);
    size_t inTensorIdx = 0;
    size_t xInTensorIdx = inTensorIdx++;
    Status st = CreateXAclnnTensor(xInTensorIdx, xInTensorIdx);
    if (st != NO_ERROR) {
        return st;
    }
    size_t weightInTensorIdx = inTensorIdx++;
    st = CreateWeightAclnnTensor(weightInTensorIdx, weightInTensorIdx);
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.hasBias) {
        size_t biasInTensorIdx = inTensorIdx++;
        st = CreateBiasAclnnTensor(biasInTensorIdx, biasInTensorIdx);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (param_.outDataType != ACL_DT_UNDEFINED) {
        size_t deqScaleInTensorIdx = inTensorIdx++;
        st = CreateDeqScaleAclnnTensor(deqScaleInTensorIdx, deqScaleInTensorIdx);
        if (st != NO_ERROR) {
            return st;
        }
    }

    aclnnVariantPack_.aclOutTensors.reserve(outTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(outTensorNum_);
    return CreateOutputAclnnTensor(0, 0);
}

aclnnStatus LinearAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::SetAclNNWorkspaceExecutor";
    if (LoadMethod() != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return 561003;  // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    }

    aclOpExecutor *raw_executor_ptr = aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(aclnnExecutor_): " << &(aclnnExecutor_)
                  << ", addr of aclnnExecutor_: " << aclnnExecutor_ << ", raw ptr from it: " << raw_executor_ptr
                  << ", then take the address of the raw ptr: " << &raw_executor_ptr;

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(atbVariantPack_.workspaceBufferSize);

    int8_t cubeMathType = 1;

    size_t inTensorIdx = 0;
    ATB_LOG(INFO) << GetLogPrefix() << "aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();
    aclTensor *x = aclnnVariantPack_.aclInTensors.at(inTensorIdx++)->tensor;
    aclTensor *weight = aclnnVariantPack_.aclInTensors.at(inTensorIdx++)->tensor;
    size_t outTensorIdx = 0;
    aclTensor *output = aclnnVariantPack_.aclOutTensors.at(outTensorIdx++)->tensor;

    aclnnStatus ret = ACL_SUCCESS;
    if (aclnnVariantPack_.aclInTensors.size() == 2) {
        ret = aclnnMatmulGetWorkspaceSizeFunc_(
            x, weight, output, cubeMathType, &(atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
        aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
            if (ptr && executorRepeatable_) {  // 可复用时才手动销毁aclOpExecutor
                aclDestroyAclOpExecutor(ptr);
            }
        });
    } else {
        aclTensor *bias = aclnnVariantPack_.aclInTensors.at(inTensorIdx++)->tensor;
        float alphaValue = 1.0f;
        float betaValue = 1.0f;
        alpha_ = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
        beta_ = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
        ret = aclnnAddmmGetWorkspaceSizeFunc_(bias,
            x,
            weight,
            beta_,
            alpha_,
            output,
            cubeMathType,
            &(atbVariantPack_.workspaceBufferSize),
            &raw_executor_ptr);
        aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
            if (ptr && executorRepeatable_) {  // 可复用时才手动销毁aclOpExecutor
                aclDestroyAclOpExecutor(ptr);
            }
        });
    }

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status LinearAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    Status st = LoadMethod();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return st;
    }
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);

    aclnnStatus ret = ACL_SUCCESS;
    if (aclnnVariantPack_.aclInTensors.size() == 2) {
        ret = aclnnMatmulExecuteFunc_(
            atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    } else {
        ret = aclnnAddmmExecuteFunc_(
            atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

void LinearAclnnRunner::GetInTensorNum()
{
    inTensorNum_ = 2;  // x, weight
    if (param_.outDataType == ACL_DT_UNDEFINED) {
        if (param_.hasBias || param_.enAccum) {
            inTensorNum_ += 1;  // bias/accum
        }
    } else {
        inTensorNum_ += 1;  // deqScale
        if (param_.hasBias) {
            inTensorNum_ += 1;  // bias
        }
        if (param_.quantMode == infer::LinearParam::PER_TOKEN) {
            inTensorNum_ += 1;  // perTokenScale
        }
    }
}

Status LinearAclnnRunner::CreateXAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex)
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateXAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbTensorIndex);
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
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "tensor" << atbTensorIndex << " aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclnnTensorIndex) = aclnnTensorPtr;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateWeightAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex)
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateWeightAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbTensorIndex);
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
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "tensor" << atbTensorIndex << " aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclnnTensorIndex) = aclnnTensorPtr;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateBiasAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex)
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateBiasAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbTensorIndex);
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
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "tensor" << atbTensorIndex << " aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclnnTensorIndex) = aclnnTensorPtr;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateDeqScaleAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex)
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateDeqScaleAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbTensorIndex);
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
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "tensor" << atbTensorIndex << " aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclnnTensorIndex) = aclnnTensorPtr;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateOutputAclnnTensor(size_t atbTensorIndex, size_t aclnnTensorIndex)
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateOutputAclnnTensor";
    Tensor atbTensor = atbVariantPack_.outTensors.at(atbTensorIndex);
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
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "tensor" << atbTensorIndex << " aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclnnTensorIndex) = aclnnTensorPtr;
    return NO_ERROR;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::InitAclnnTensor(Tensor atbTensor, size_t aclnnTensorIndex)
{
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->tensorIdx = static_cast<int>(aclnnTensorIndex);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    return aclnnTensorPtr;
}

REG_RUNNER_TYPE(LinearAclnnRunner);
}  // namespace atb
