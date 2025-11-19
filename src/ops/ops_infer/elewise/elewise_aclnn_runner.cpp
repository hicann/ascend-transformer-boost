/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "elewise_aclnn_runner.h"
#include <aclnn/opdev/bfloat16.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/dl_manager.h"
#include "atb/utils/utils_internal.h"

static const uint32_t IN_TENSOR_NUM = 3;
static const uint32_t IN_TENSOR_IDX = 0;
static const uint32_t SCALE_TENSOR_IDX = 1;
static const uint32_t OFFSET_TENSOR_IDX = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_IDX = 0;
// 指定scale参与计算的逻辑
static const bool SQRT_MODE = false;
// 指定cast到int8输出的转换方式。支持取值round/ceil/trunc/floor。
static const std::string ROUND_MODE = "round";
// 指定`scale`和`offset`对应`x`的维度。当输入`x`的数据格式为NZ时，取值为-1。
static const int AXIS = -1;

namespace atb {
AclnnAscendQuantGetWsFunc ElewiseAclnnRunner::aclnnAscendQuantGetWorkspaceSizeFunc_ = nullptr;
AclnnAscendQuantExecFunc ElewiseAclnnRunner::aclnnAscendQuantExecuteFunc_ = nullptr;

ElewiseAclnnRunner::KernelAdapters ElewiseAclnnRunner::MakeAdaptersByType(const infer::ElewiseParam &param) {
    switch (param.elewiseType) {
        case infer::ElewiseParam::ElewiseType::ELEWISE_QUANT:
        return {
            "quant",
            [](const AclNNVariantPack &vp, const infer::ElewiseParam &param, uint64_t *wsSize, aclOpExecutor **executor) {
                if (!aclnnAscendQuantGetWorkspaceSizeFunc_) {
                    ATB_LOG(ERROR) << "aclnnAscendQuantGetWorkspaceSizeFunc_ is not initialized";
                    return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
                }
                return aclnnAscendQuantGetWorkspaceSizeFunc_(vp.aclInTensors.at(0)->tensor,
                                                             vp.aclInTensors.at(1)->tensor,
                                                             vp.aclInTensors.at(2)->tensor,
                                                             SQRT_MODE,
                                                             ROUND_MODE.c_str(),
                                                             param.outTensorType,
                                                             AXIS,
                                                             vp.aclOutTensors.at(0)->tensor,
                                                             wsSize,
                                                             executor);
            },
            [](void *ws, uint64_t wsSize, aclOpExecutor *executor, aclrtStream stream) {
                if (!aclnnAscendQuantExecuteFunc_) {
                    ATB_LOG(ERROR) << "aclnnAscendQuantExecuteFunc_ is not initialized";
                    return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
                }
                return aclnnAscendQuantExecuteFunc_(ws, wsSize, executor, stream);
            }
        };

        default:
        return {
            "unsupported",
            [](const AclNNVariantPack &, const infer::ElewiseParam &, uint64_t *, aclOpExecutor **) {
                return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
            },
            [](void *, uint64_t, aclOpExecutor *, aclrtStream) {
                return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
            }
        };
    }
}

ElewiseAclnnRunner::ElewiseAclnnRunner(const infer::ElewiseParam &param)
    : AclnnRunner("ElewiseAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ElewiseAclnnRunner called, Quant type: " << param_.elewiseType;
}

ElewiseAclnnRunner::~ElewiseAclnnRunner()
{
    if (this->scaleTensor.hostData != nullptr) {
        free(this->scaleTensor.hostData);
        this->scaleTensor.hostData = nullptr;
    }

    if (this->scaleTensor.deviceData != nullptr) {
        aclrtFree(this->scaleTensor.deviceData);
        this->scaleTensor.deviceData = nullptr;
    }

    if (this->offsetTensor.hostData != nullptr) {
        free(this->offsetTensor.hostData);
        this->offsetTensor.hostData = nullptr;
    }

    if (this->offsetTensor.deviceData != nullptr) {
        aclrtFree(this->offsetTensor.deviceData);
        this->offsetTensor.deviceData = nullptr;
    }
}

Status ElewiseAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack called.";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);

    ATB_LOG(INFO) << GetLogPrefix() << "Processing inTensor...";
    std::shared_ptr<AclNNTensor> xTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor xTensor = runnerVariantPack.inTensors.at(IN_TENSOR_IDX);

    xTensorPtr->atbTensor = xTensor;
    xTensorPtr->strides = GetCopyTensorStride(xTensor.desc.shape);

    ret = CallAclCreateTensor(xTensor.desc.shape, xTensor.desc.shape, xTensor, xTensorPtr,
                              xTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    xTensorPtr->tensorIdx = static_cast<int>(IN_TENSOR_IDX);
    xTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclInTensors[IN_TENSOR_IDX] = xTensorPtr;

    // set dataSize
    uint64_t totalDataSize = UtilsInternal::GetDataTypeSize(xTensor.desc.dtype);

    // set scale
    std::shared_ptr<AclNNTensor> scaleTensorPtr = std::make_shared<AclNNTensor>();
    this->scaleTensor.desc.dtype = xTensor.desc.dtype; // 如果x的数据类型不是FLOAT32，数据类型需要和`x`的数据类型一致。
    this->scaleTensor.desc.format = ACL_FORMAT_ND;
    this->scaleTensor.dataSize = totalDataSize;
    this->scaleTensor.desc.shape.dimNum = 1;
    this->scaleTensor.desc.shape.dims[0] = 1;

    this->scaleTensor.hostData = nullptr;
    if (xTensor.desc.dtype == ACL_FLOAT16) {
        this->scaleTensor.hostData = malloc(sizeof(aclFloat16));
        *((aclFloat16*) this->scaleTensor.hostData) = aclFloatToFloat16(param_.quantParam.inputScale);
    } else if (xTensor.desc.dtype == ACL_FLOAT) {
        this->scaleTensor.hostData = malloc(sizeof(float));
        *((float*) this->scaleTensor.hostData) = param_.quantParam.inputScale;
    } else if (xTensor.desc.dtype == ACL_BF16) {
        this->scaleTensor.hostData = malloc(sizeof(op::bfloat16));
        *((op::bfloat16*) this->scaleTensor.hostData) = static_cast<op::bfloat16>(param_.quantParam.inputScale);
    }
    if (this->scaleTensor.hostData == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "malloc scaleTensor.hostData failed.";
        return ERROR_INTERNAL_ERROR;
    }
    auto acl_ret = aclrtMalloc(&this->scaleTensor.deviceData, this->scaleTensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclrtMalloc failed. ERROR: " << acl_ret;
        return acl_ret;
    }
    acl_ret = aclrtMemcpy(this->scaleTensor.deviceData, this->scaleTensor.dataSize, this->scaleTensor.hostData, this->scaleTensor.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_ret != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclrtMemcpy failed. ERROR: " << acl_ret;
        return acl_ret;
    }
    scaleTensorPtr->atbTensor = this->scaleTensor;
    scaleTensorPtr->strides = GetCopyTensorStride(this->scaleTensor.desc.shape);
    ret = CallAclCreateTensor(this->scaleTensor.desc.shape, this->scaleTensor.desc.shape, this->scaleTensor, scaleTensorPtr,
                                this->scaleTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    scaleTensorPtr->tensorIdx = static_cast<int>(SCALE_TENSOR_IDX);
    scaleTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclInTensors[SCALE_TENSOR_IDX] = scaleTensorPtr;

    // set offset
    std::shared_ptr<AclNNTensor> offsetTensorPtr = std::make_shared<AclNNTensor>();
    this->offsetTensor.desc.dtype = this->scaleTensor.desc.dtype; // 与scale保持一致
    this->offsetTensor.desc.format = ACL_FORMAT_ND;
    this->offsetTensor.dataSize = totalDataSize;
    this->offsetTensor.desc.shape.dimNum = 1;
    this->offsetTensor.desc.shape.dims[0] = 1;

    this->offsetTensor.hostData = nullptr;
    if (xTensor.desc.dtype == ACL_FLOAT16) {
        this->offsetTensor.hostData = malloc(sizeof(aclFloat16));
        *((aclFloat16*) this->offsetTensor.hostData) = aclFloatToFloat16(static_cast<float>(param_.quantParam.inputOffset));
    } else if (xTensor.desc.dtype == ACL_FLOAT) {
        this->offsetTensor.hostData = malloc(sizeof(float));
        *((float*) this->offsetTensor.hostData) = static_cast<float>(param_.quantParam.inputOffset);
    } else if (xTensor.desc.dtype == ACL_BF16) {
        this->offsetTensor.hostData = malloc(sizeof(op::bfloat16));
        *((op::bfloat16*) this->offsetTensor.hostData) = static_cast<op::bfloat16>(param_.quantParam.inputOffset);
    }
    if (this->offsetTensor.hostData == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "malloc offsetTensor.hostData failed.";
        return ERROR_INTERNAL_ERROR;
    }

    acl_ret = aclrtMalloc(&this->offsetTensor.deviceData, this->offsetTensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != 0) {
        ATB_LOG(ERROR) << "aclrtMalloc failed. ERROR: " << acl_ret;
        return acl_ret;
    }
    acl_ret = aclrtMemcpy(this->offsetTensor.deviceData, this->offsetTensor.dataSize, this->offsetTensor.hostData, this->offsetTensor.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_ret != 0) {
        ATB_LOG(ERROR) << "aclrtMemcpy failed. ERROR: " << acl_ret;
        return acl_ret;
    }
    offsetTensorPtr->atbTensor = this->offsetTensor;
    offsetTensorPtr->strides = GetCopyTensorStride(this->offsetTensor.desc.shape);
    ret = CallAclCreateTensor(this->offsetTensor.desc.shape, this->offsetTensor.desc.shape, this->offsetTensor, offsetTensorPtr,
                                this->offsetTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    offsetTensorPtr->tensorIdx = static_cast<int>(OFFSET_TENSOR_IDX);
    offsetTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclInTensors[OFFSET_TENSOR_IDX] = offsetTensorPtr;

    // set out tensor
    ATB_LOG(INFO) << GetLogPrefix() << "Processing outTensor...";
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    std::shared_ptr<AclNNTensor> yTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor yTensor = runnerVariantPack.outTensors.at(OUT_TENSOR_IDX);
    yTensorPtr->atbTensor = yTensor;
    yTensorPtr->strides = GetCopyTensorStride(yTensor.desc.shape);
    ret = CallAclCreateTensor(yTensor.desc.shape, yTensor.desc.shape, yTensor, yTensorPtr);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    yTensorPtr->tensorIdx = static_cast<int>(OUT_TENSOR_IDX);
    yTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclOutTensors[OUT_TENSOR_IDX] = yTensorPtr;

    return ret;
}

aclnnStatus ElewiseAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Quant setup start.";
    Status status = ElewiseAclnnRunner::LoadAclnnFunctions();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "load getWorkspaceSize and execute func failed.";
        return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    }
    KernelAdapters kernelAdapter = MakeAdaptersByType(param_);
    this->adapterGetWs_ = std::move(kernelAdapter.getWs);
    this->adapterExec_ = std::move(kernelAdapter.exec);
    if (!adapterGetWs_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Alcnn GetWorkspaceSizeFunc is null!";
        return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    }

    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Quant type: " << param_.elewiseType
                  << ", aclInTensor size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensor size: " << this->aclnnVariantPack_.aclOutTensors.size();

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    ATB_LOG(DEBUG) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << raw_executor_ptr
                  << ", then take the address of the raw ptr: " << &raw_executor_ptr;

    ATB_LOG(DEBUG) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    aclnnStatus ret = this->adapterGetWs_(this->aclnnVariantPack_, param_, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnGetWorkspaceSizeFunc failed, error: " << ret;
        return ret;
    }
    this->aclnnExecutor_.reset(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });

    ATB_LOG(INFO) << GetLogPrefix() << "WorkspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status ElewiseAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel called.";
    Status status = ElewiseAclnnRunner::LoadAclnnFunctions();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Load getWorkspaceSize and execute func from so failed.";
        return status;
    }
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = this->adapterExec_(this->atbVariantPack_.workspaceBuffer,
                                    this->atbVariantPack_.workspaceBufferSize,
                                    this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Launch kernel failed with error: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel success.";
    return NO_ERROR;
}

Status ElewiseAclnnRunner::LoadAclnnFunctions()
{
    ATB_LOG(INFO) << "ElewiseAclnnRunner LoadAclnnFunctions...";
    if (aclnnAscendQuantGetWorkspaceSizeFunc_ && aclnnAscendQuantExecuteFunc_) {
        return NO_ERROR;
    }
    static DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");

    Status ret =
        dlManager.getSymbol("aclnnAscendQuantV3GetWorkspaceSize", (void *&)aclnnAscendQuantGetWorkspaceSizeFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnAscendQuantV3GetWorkspaceSize failed! Consider upgrade the CANN first!";
        return ret;
    }
    ret = dlManager.getSymbol("aclnnAscendQuantV3", (void *&)aclnnAscendQuantExecuteFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnAscendQuantV3 failed! Consider upgrade the CANN first!";
        return ret;
    }

    ATB_LOG(INFO) << "load aclnnAscendQuantV3 two-staged method success!";
    return NO_ERROR;
}

REG_RUNNER_TYPE(ElewiseAclnnRunner);
} // namespace atb