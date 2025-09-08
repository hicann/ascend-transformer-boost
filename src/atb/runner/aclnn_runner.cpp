/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/aclnn_runner.h"
#include "atb/core/aclnn/aclnn_executor_cache.h"
#include "atb/core/aclnn/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/singleton.h"

namespace atb {

AclnnRunner::AclnnRunner(const std::string &name, RunnerType runnerType) : Runner(name), runnerType_(runnerType){}

AclnnRunner::~AclnnRunner() {}

Status AclnnRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!runnerVariantPack.context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is not ContextBase, setup fail";
        return ERROR_INVALID_CONTEXT_ADDR;
    }
    aclError ret = ACL_SUCCESS;
    if (!executorRepeatable_) {
        ret = SetAclNNWorkspaceExecutor();
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << "Atb aclnn op set workspace failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
    }
    // variantPack与上次对应，直接使用
    if (IsAclnnRunnerVariankPackEqual(this->aclnnVariantPack_, runnerVariantPack)) {
        return NO_ERROR;
    }
    const std::string &opName = this->GetName();
    AclnnCacheSlot aclnnCacheSlot = {};
    // executorCache hit
    if (GetSingleton<AclnnExecutorCache>().FetchCacheSlot(opName, runnerVariantPack, aclnnCacheSlot) == NO_ERROR) {
        this->atbVariantPack_ = runnerVariantPack;
        // TODO：更新this->aclNNVariantPack_
        BuildAclnnVariantPack(runnerVariantPack);
        this->workspaceSize_ = aclnnCacheSlot.workspaceSize;
        this->aclnnExecutor_ = aclnnCacheSlot.executor;
        return NO_ERROR;
    }
    // 创建新的executor
    BuildAclnnVariantPack(runnerVariantPack);
    ret = SetAclNNWorkspaceExecutor();
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << "Atb aclnn op set workspace failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    aclnnCacheSlot = {this->workspaceSize_, aclnnExecutor_};
    return GetSingleton<AclnnExecutorCache>().AddCacheSlot(opName, runnerVariantPack, aclnnCacheSlot);
}

uint64_t AclnnRunner::GetWorkspaceBufferSizeImpl()
{
    return workspaceSize_;
}

Status AclnnRunner::PreExecuteImpl(RunnerVariantPack &runnerVariantPack) {
    ATB_LOG(INFO) << "AclNNOpCacheUpdateAclNNVariantPack";
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        int ret = -1;
        if (!this->aclnnVariantPack_.aclInTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack_.aclInTensors[i]->atbTensor = runnerVariantPack.inTensors.at(i);
        if (this->aclnnVariantPack_.aclInTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetInputTensorAddr(this->aclnnExecutor_.get(), this->aclnnVariantPack_.aclInTensors[i]->tensorIdx,
                                        this->aclnnVariantPack_.aclInTensors[i]->tensor,
                                        this->aclnnVariantPack_.aclInTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicInputTensorAddr(
                this->aclnnExecutor_.get(), this->aclnnVariantPack_.aclInTensors[i]->tensorListidx,
                this->aclnnVariantPack_.aclInTensors[i]->tensorIdx,
                this->aclnnVariantPack_.aclInTensorList[this->aclnnVariantPack_.aclInTensors[i]->tensorListidx],
                this->aclnnVariantPack_.aclInTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << "inTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        int ret = -1;
        if (!this->aclnnVariantPack_.aclOutTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack_.aclOutTensors[i]->atbTensor = runnerVariantPack.outTensors.at(i);
        if (this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetOutputTensorAddr(this->aclnnExecutor_.get(), this->aclnnVariantPack_.aclOutTensors[i]->tensorIdx,
                                         this->aclnnVariantPack_.aclOutTensors[i]->tensor,
                                         this->aclnnVariantPack_.aclOutTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicOutputTensorAddr(
                this->aclnnExecutor_.get(), this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx,
                this->aclnnVariantPack_.aclOutTensors[i]->tensorIdx,
                this->aclnnVariantPack_.aclOutTensorList[this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx],
                this->aclnnVariantPack_.aclOutTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << "outTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    return atb::NO_ERROR;
}

Status AclnnRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack) {
    Status ret = BuildAclnnVariantPack(runnerVariantPack);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "ATB aclnn runner: BuildAclnnVariantPack error!";
        return ret;
    }
    return LaunchAclnnKernel(this->aclnnVariantPack_);
}

} // namespace atb