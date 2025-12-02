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

AclnnGetWorkspaceSizeFunc RmsNormAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
AclnnExecuteFunc RmsNormAclnnRunner::aclnnExecuteFunc_ = nullptr;

RmsNormAclnnRunner::RmsNormAclnnRunner(const infer::RmsNormParam &param)
    : AclnnRunner("RmsNormAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormAclnnRunner::RmsNormAclnnRunner called";
}

RmsNormAclnnRunner::~RmsNormAclnnRunner()
{
    aclnnStatus ret = aclDestroyTensor(rstdTensor_);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "destroy scale tensor ERROR: " << ret;
    }
    aclError err = aclrtFree(rstdDeviceAddr_);
    if (err != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "free scale device addr ERROR: " << err;
    }
}

Status RmsNormAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "RmsNormAclnnRunner LoadMethod";
    Status status = NO_ERROR;
    if (aclnnGetWorkspaceSizeFunc_ == nullptr || aclnnExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnRmsNormGetWorkspaceSize", "aclnnRmsNorm",
                                           aclnnGetWorkspaceSizeFunc_,
                                           aclnnExecuteFunc_);
    }
    return status;
}

Status RmsNormAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack, runnerVariantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    ret = CreateInputAclnnTensor();
    if (ret != NO_ERROR) {
        return ret;
    }
    ret = CreateGammaAclnnTensor();
    if (ret != NO_ERROR) {
        return ret;
    }
    aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    ret = CreateOutputAclnnTensor();
    if (ret != NO_ERROR) {
        return ret;
    }
    ret = CreateRstdAclnnTensor();
    if (ret != NO_ERROR) {
        return ret;
    }
    return ret;
}

Status RmsNormAclnnRunner::CreateInputAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormAclnnRunner::CreateInputAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor atbTensor = atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO);
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);

    Status ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor,
                                     aclnnTensorPtr, atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create input aclTensor by aclCreateTensor failed!";
        return ret;
    }

    aclnnTensorPtr->tensorIdx = static_cast<int>(TENSOR_IDX_ZERO);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    aclnnVariantPack_.aclInTensors.at(TENSOR_IDX_ZERO) = aclnnTensorPtr;

    return NO_ERROR;
}

Status RmsNormAclnnRunner::CreateGammaAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormAclnnRunner::CreateGammaAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor atbTensor = atbVariantPack_.inTensors.at(IN_TENSOR_NUM - 1);

    // 对gamma张量进行特殊处理：压缩维度中的1
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

    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);

    Status ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor,
                                     aclnnTensorPtr, atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create gamma aclTensor by aclCreateTensor failed!";
        return ret;
    }

    aclnnTensorPtr->tensorIdx = static_cast<int>(IN_TENSOR_NUM - 1);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    aclnnVariantPack_.aclInTensors.at(IN_TENSOR_NUM - 1) = aclnnTensorPtr;

    return NO_ERROR;
}

Status RmsNormAclnnRunner::CreateOutputAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormAclnnRunner::CreateOutputAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor atbTensor = atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO); // 使用输入张量的形状

    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);

    Status ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor,
                                     aclnnTensorPtr, atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create output aclTensor by aclCreateTensor failed!";
        return ret;
    }

    aclnnTensorPtr->tensorIdx = static_cast<int>(TENSOR_IDX_ZERO);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    aclnnVariantPack_.aclOutTensors.at(TENSOR_IDX_ZERO) = aclnnTensorPtr;

    return NO_ERROR;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

aclnnStatus  RmsNormAclnnRunner::CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor, uint64_t dataSize)
{
	aclnnStatus ret = ACL_SUCCESS;
    ret = aclrtMalloc(deviceAddr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclrtMalloc failed. ERROR: " << ret;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return ret;
}

Status RmsNormAclnnRunner::CreateRstdAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormAclnnRunner::CreateRstdAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();

    if (param_.normParam.rstd) {
        atb::Tensor atbTensor = atbVariantPack_.outTensors.at(TENSOR_IDX_ONE);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);

        Status ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor,
                                     aclnnTensorPtr, atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create rstd aclTensor by aclCreateTensor failed!";
            return ret;
        }
    } else {
        std::vector<int64_t> shape;
        shape.reserve(atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO).desc.shape.dimNum);
        for (size_t i = 0; i < atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO).desc.shape.dimNum; ++i) {
            shape.push_back(atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO).desc.shape.dims[i]);
        }

        uint64_t dataSize =  atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO).dataSize;

        if (atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO).desc.dtype == ACL_FLOAT) {
            dataSize /= DATASIZE_32BIT;
        } else {
            dataSize /= DATASIZE_16BIT;
        }
        dataSize *= DATASIZE_32BIT;

        for (size_t i = 0; i < atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO).desc.shape.dimNum; ++i) {
            if (i >= atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO).desc.shape.dimNum -
                atbVariantPack_.inTensors.at(TENSOR_IDX_ONE).desc.shape.dimNum) {
                dataSize /= shape[i];
                shape[i] = 1;
            } else {
                shape[i] = atbVariantPack_.inTensors.at(TENSOR_IDX_ZERO).desc.shape.dims[i];
            }
        }

        aclnnStatus ret = CreateAclTensor(shape, &rstdDeviceAddr_, aclDataType::ACL_FLOAT, &rstdTensor_, dataSize);
		if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "CreateAclTensor failed! " << ret;
            return ret;
        }
        aclnnTensorPtr->tensor = rstdTensor_;
    }

    aclnnVariantPack_.aclOutTensors.at(TENSOR_IDX_ONE) = aclnnTensorPtr;
    return NO_ERROR;
}

Status RmsNormAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = RmsNormAclnnRunner::aclnnExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                                                  atbVariantPack_.workspaceBufferSize,
                                                                  aclnnExecutor_.get(), executeStream);
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
    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();
    aclTensor *x = aclnnVariantPack_.aclInTensors.at(0)->tensor;
    aclTensor *gamma = aclnnVariantPack_.aclInTensors.at(1)->tensor;
    aclTensor *yOut = aclnnVariantPack_.aclOutTensors.at(0)->tensor;
    aclTensor *rstd = aclnnVariantPack_.aclOutTensors.at(1)->tensor;
    double epsilon = (double)param_.normParam.epsilon;
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(aclnnExecutor_): " << &(aclnnExecutor_)
                  << ", addr of aclnnExecutor_: " << aclnnExecutor_
                  << ", raw ptr from it: " << rawExecutorPtr
                  << ", then take the address of the raw ptr: " << &rawExecutorPtr;
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(atbVariantPack_.workspaceBufferSize);
    aclnnStatus ret = RmsNormAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        x,
        gamma,
        epsilon,
        yOut,
        rstd,
        &(atbVariantPack_.workspaceBufferSize),
        &rawExecutorPtr
    );
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}



REG_RUNNER_TYPE(RmsNormAclnnRunner);
} // namespace atb