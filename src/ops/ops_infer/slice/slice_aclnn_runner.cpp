/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "slice_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
} // namespace

namespace atb {
AclnnSliceV2GetWorkspaceSizeFunc SliceAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
AclnnSliceV2Func SliceAclnnRunner::aclnnExecuteFunc_ = nullptr;

SliceAclnnRunner::SliceAclnnRunner(const infer::SliceParam &param) : AclnnRunner("SliceAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SliceAclnnRunner::SliceAclnnRunner created";
}

SliceAclnnRunner::~SliceAclnnRunner() {}

Status SliceAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "SliceAclnnRunner LoadAclnnFuncs";
    if (SliceAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr && SliceAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnSliceV2GetWorkspaceSize", "aclnnSliceV2",
                                    SliceAclnnRunner::aclnnGetWorkspaceSizeFunc_, SliceAclnnRunner::aclnnExecuteFunc_);
}

Status SliceAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // self
    aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < aclnnVariantPack_.aclInTensors.size(); ++i) {
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
        aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    // output
    aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack outTensor index: " << i;
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
        aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    return atb::NO_ERROR;
}

aclnnStatus SliceAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Slice setup start.";
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Slice, aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();
    std::shared_ptr<AclNNTensor> xPtr = aclnnVariantPack_.aclInTensors.at(0);
    aclTensor *x = xPtr->tensor;                                       // self
    aclTensor *output = aclnnVariantPack_.aclOutTensors.at(0)->tensor; // out
    Dims xDims = xPtr->atbTensor.desc.shape;

    SVector<int64_t> offsets = param_.offsets;
    SVector<int64_t> size = param_.size;
    int64_t dimNum = offsets.size();
    int64_t steps[dimNum];
    int64_t axes[dimNum];
    int64_t starts[dimNum];
    int64_t ends[dimNum];
    for (int64_t i = 0; i < dimNum; ++i) {
        steps[i] = 1;
        axes[i] = i;
        starts[i] = offsets[i];
        if (offsets[i] < 0) {
            starts[i] = xDims.dims[i] + offsets[i];
        } else {
            starts[i] = offsets[i];
        }
        if (size[i] == -1) {
            ends[i] = xDims.dims[i];
        } else {
            ends[i] = starts[i] + size[i];
        }
    }

    aclnnStatus ret = ACL_SUCCESS;
    if (stepsArray_) {
        ret = aclDestroyIntArray(stepsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray steps failed!";
            return ret;
        }
        stepsArray_ = nullptr;
    }
    if (axesArray_) {
        ret = aclDestroyIntArray(axesArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray axes failed!";
            return ret;
        }
        axesArray_ = nullptr;
    }
    if (startsArray_) {
        ret = aclDestroyIntArray(startsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray starts failed!";
            return ret;
        }
        startsArray_ = nullptr;
    }
    if (endsArray_) {
        ret = aclDestroyIntArray(endsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray ends failed!";
            return ret;
        }
        startsArray_ = nullptr;
    }
    stepsArray_ = aclCreateIntArray(steps, dimNum);
    axesArray_ = aclCreateIntArray(axes, dimNum);
    startsArray_ = aclCreateIntArray(starts, dimNum);
    endsArray_ = aclCreateIntArray(ends, dimNum);

    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();

    ret = SliceAclnnRunner::aclnnGetWorkspaceSizeFunc_(x, startsArray_, endsArray_, axesArray_, stepsArray_, output,
                                                       &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status SliceAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    void *executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = SliceAclnnRunner::aclnnExecuteFunc_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    if (stepsArray_) {
        ret = aclDestroyIntArray(stepsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray steps failed!";
            return ERROR_CANN_ERROR;
        }
        stepsArray_ = nullptr;
    }
    if (axesArray_) {
        ret = aclDestroyIntArray(axesArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray axes failed!";
            return ERROR_CANN_ERROR;
        }
        axesArray_ = nullptr;
    }
    if (startsArray_) {
        ret = aclDestroyIntArray(startsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray starts failed!";
            return ERROR_CANN_ERROR;
        }
        startsArray_ = nullptr;
    }
    if (endsArray_) {
        ret = aclDestroyIntArray(endsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray ends failed!";
            return ERROR_CANN_ERROR;
        }
        endsArray_ = nullptr;
    }
    return NO_ERROR;
}

REG_RUNNER_TYPE(SliceAclnnRunner);
} // namespace atb
