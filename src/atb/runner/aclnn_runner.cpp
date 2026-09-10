/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/aclnn_runner.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/singleton.h"
#include "atb/utils/operation_register.h"

namespace atb {

AclnnRunner::AclnnRunner(const std::string &name) : Runner(name)
{
    runnerTypeIdx_ = RunnerTypeRegister::GetRunnerTypeIdx(name);
}

AclnnRunner::~AclnnRunner()
{
    for (size_t i = 0; i < aclnnVariantPack_.aclInTensorList.size(); i++) {
        if (aclnnVariantPack_.aclInTensorList.at(i)) {
            if (aclDestroyTensorList(aclnnVariantPack_.aclInTensorList.at(i)) != ACL_SUCCESS) {
                ATB_LOG(ERROR) << "aclInTensorList[" << i << "] aclDestroyTensorList failed";
            }
            aclnnVariantPack_.aclInTensorList.at(i) = nullptr;
        }
    }
    for (size_t i = 0; i < aclnnVariantPack_.aclOutTensorList.size(); i++) {
        if (aclnnVariantPack_.aclOutTensorList.at(i)) {
            if (aclDestroyTensorList(aclnnVariantPack_.aclOutTensorList.at(i)) != ACL_SUCCESS) {
                ATB_LOG(ERROR) << "aclOutTensorList[" << i << "] aclDestroyTensorList failed";
            }
            aclnnVariantPack_.aclOutTensorList.at(i) = nullptr;
        }
    }
    aclnnVariantPack_.aclInTensorList.clear();
    aclnnVariantPack_.aclOutTensorList.clear();
    aclnnVariantPack_.aclInTensors.clear();
    aclnnVariantPack_.aclOutTensors.clear();
}

Status AclnnRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn runner setupImpl";
    if (!runnerVariantPack.context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is not ContextBase, setup fail";
        return ERROR_INVALID_CONTEXT_ADDR;
    }

    if (executorRepeatable_) {
        ATB_LOG(INFO) << GetLogPrefix() << "Setup reuse branch";
        if (IsAclnnRunnerVariankPackEqual(this->aclnnVariantPack_, runnerVariantPack)) {
            ATB_LOG(INFO) << GetLogPrefix() << "Setup reuse return";
            return NO_ERROR;
        }
        ATB_LOG(INFO) << GetLogPrefix()
                      << "fetched cached runnerVariantPack not same as aclnnVariantPack_, build again";
    }

    executorRepeatable_ = false;
    Status ret = BuildAclnnVariantPack(runnerVariantPack);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "BuildAclnnVariantPack failed!";
        return ret;
    }
    aclnnStatus aclnnRet = SetAclNNWorkspaceExecutor();
    if (aclnnRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op set workspace failed with return value: " << aclnnRet;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "Setup update executor, repeatable: " << executorRepeatable_;
    ATB_LOG(INFO) << GetLogPrefix()
                  << "getWorkspaceSize success, workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

uint64_t AclnnRunner::GetWorkspaceBufferSizeImpl()
{
    return this->atbVariantPack_.workspaceBufferSize;
}

Status AclnnRunner::PreExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AclNNOpCacheUpdateAclNNVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "PreExecute update tensor addresses";
    aclnnStatus ret = ACL_SUCCESS;
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        // 部分场景中存在aclnn接口使用空tensor占位最后可选tensor，但是runnerVariantPack中不存放tensor的情况，可以跳过
        if (i >= runnerVariantPack.inTensors.size()) {
            break;
        }
        if (this->aclnnVariantPack_.aclInTensors[i] == nullptr ||
            !this->aclnnVariantPack_.aclInTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack_.aclInTensors[i]->atbTensor = runnerVariantPack.inTensors.at(i);
        if (this->aclnnVariantPack_.aclInTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetInputTensorAddr(this->atbAclOpExecutor_->Get(),
                                        this->aclnnVariantPack_.aclInTensors[i]->tensorIdx,
                                        this->aclnnVariantPack_.aclInTensors[i]->tensor,
                                        this->aclnnVariantPack_.aclInTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicInputTensorAddr(
                this->atbAclOpExecutor_->Get(), this->aclnnVariantPack_.aclInTensors[i]->tensorListidx,
                this->aclnnVariantPack_.aclInTensors[i]->tensorIdx,
                this->aclnnVariantPack_.aclInTensorList[this->aclnnVariantPack_.aclInTensors[i]->tensorListidx],
                this->aclnnVariantPack_.aclInTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        if (i >= runnerVariantPack.outTensors.size()) {
            break;
        }
        if (this->aclnnVariantPack_.aclOutTensors[i] == nullptr ||
            !this->aclnnVariantPack_.aclOutTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack_.aclOutTensors[i]->atbTensor = runnerVariantPack.outTensors.at(i);
        if (this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetOutputTensorAddr(this->atbAclOpExecutor_->Get(),
                                         this->aclnnVariantPack_.aclOutTensors[i]->tensorIdx,
                                         this->aclnnVariantPack_.aclOutTensors[i]->tensor,
                                         this->aclnnVariantPack_.aclOutTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicOutputTensorAddr(
                this->atbAclOpExecutor_->Get(), this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx,
                this->aclnnVariantPack_.aclOutTensors[i]->tensorIdx,
                this->aclnnVariantPack_.aclOutTensorList[this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx],
                this->aclnnVariantPack_.aclOutTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor " << i
                           << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    return atb::NO_ERROR;
}

void AclnnRunner::UpdateWorkspace(const RunnerVariantPack &runnerVariantPack)
{
    this->atbVariantPack_.workspaceBufferSize = runnerVariantPack.workspaceBufferSize;
    this->atbVariantPack_.workspaceBuffer = runnerVariantPack.workspaceBuffer;
}

Status AclnnRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AclnnRunner::ExecuteImpl";
    UpdateWorkspace(runnerVariantPack);
    return LaunchAclnnKernel();
}

} // namespace atb
