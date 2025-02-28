/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/runner.h"
#include <sstream>
#include <mki/utils/file_system/file_system.h>
#include <mki/utils/time/timer.h>
#include "atb/utils/config.h"
#include "atb/utils/store_util.h"
#include "atb/utils/tensor_util.h"
#include "atb/operation/operation_base.h"
#include "atb/operation/graph_operation.h"
#include "atb/types.h"
#include "atb/utils/common_utils.h"
#include "atb/utils/log.h"
#include "atb/utils/probe.h"
#include "atb/utils/statistic.h"
#include "atb/utils/singleton.h"

namespace atb {
static const char *TENSOR_FILE_NAME_EXT = ".bin";

Runner::Runner(const std::string &name) : name_(name)
{
    tensorDir_ = name_;
    logPrefix_ = name_;
    operationName_ = "";
    runnerIds_.clear();
}

Runner::~Runner() {}

std::string Runner::GetName() const
{
    return name_;
}

std::string Runner::GetOperationName() const
{
    return operationName_;
}

Status Runner::Setup(RunnerVariantPack &runnerVariantPack)
{
    Status st = SetupImpl(runnerVariantPack);
    setupCount_++;
    return st;
}

uint64_t Runner::GetIntermediateBufferSize()
{
    return GetIntermediateBufferSizeImpl();
}

uint64_t Runner::GetTilingBufferSize()
{
    return GetTilingBufferSizeImpl();
}

Status Runner::FillHostTilingBuffer(uint8_t *hostTilingBuffer, uint64_t tilingBufferSize)
{
    return FillHostTilingBufferImpl(hostTilingBuffer, tilingBufferSize);
}

uint64_t Runner::GetWorkspaceBufferSize()
{
    return GetWorkspaceBufferSizeImpl();
}

Status Runner::Execute(RunnerVariantPack &runnerVariantPack)
{
    if (!runnerVariantPack.context) {
        ATB_LOG(ERROR) << GetLogPrefix() << " runnerVariantPack.context is null";
        return ERROR_INVALID_PARAM;
    }
    if (runnerVariantPack.context->GetExecuteStream() == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << " runnerVariantPack.stream is null";
        return ERROR_INVALID_PARAM;
    }
    if (IsSaveTensor() && Probe::IsSaveTensorBefore()) {
        std::string tensorDir = tensorDir_ + "/before";
        StoreUtil::SaveVariantPack(runnerVariantPack.context->GetExecuteStream(), runnerVariantPack, tensorDir);
        ATB_LOG(INFO) << GetLogPrefix() << " save variant pack at " << tensorDir;
    }
    Status st = ExecuteImpl(runnerVariantPack);
    if (IsSaveTensor() && Probe::IsSaveTensorAfter()) {
        std::string tensorDir = tensorDir_ + "/after";
        StoreUtil::SaveVariantPack(runnerVariantPack.context->GetExecuteStream(), runnerVariantPack, tensorDir);
        ATB_LOG(INFO) << GetLogPrefix() << " save variant pack at " << tensorDir;
    }
    if (IsSaveTensor() && Probe::IsSaveParam()) {
        OperationBase *opBase = dynamic_cast<OperationBase *>(operation_);
        if (opBase) {
            nlohmann::json opParamJson = opBase->GetParamJson();
            if (!opParamJson.empty()) {
                std::string opParam = opParamJson.dump();
                std::string filePath = tensorDir_ + "/op_param.json";
                Probe::SaveParam(opParam, filePath);
                ATB_LOG(INFO) << GetLogPrefix() << " save param at " << filePath;
            }
        }
    }
    GraphOperation *opGraph = dynamic_cast<GraphOperation *>(operation_);
    if (Probe::ReportOperationIOTensorEnable() && !opGraph) {
        DumpIOTensorInfo(runnerVariantPack);
    }
    executeCount_++;
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Execute Failed. st: " << st;
        return st;
    }
    if (GetSingleton<Config>().IsStreamSyncEveryRunnerEnable()) {
        Mki::Timer streamSyncTimer;
        int retCode = aclrtSynchronizeStream(runnerVariantPack.context->GetExecuteStream());
        GetOpExecuteStatistic().streamSyncTime += streamSyncTimer.ElapsedMicroSecond();
        ATB_LOG_IF(retCode != 0, ERROR) << GetLogPrefix() << "stream sync fail, ret:" << retCode;
    }
    return NO_ERROR;
}

void Runner::SetRunnerInfo(const std::string &operationName, const std::vector<int64_t> &operationIds)
{
    operationName_ = operationName;
    runnerIds_ = operationIds;
    logPrefix_ = GenerateOperationName(name_, runnerIds_);
    saveTensorFlag_ = Probe::IsTensorNeedSave(runnerIds_, operationName_);
}

Status Runner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack:" << runnerVariantPack.ToString();
    return NO_ERROR;
}

uint64_t Runner::GetIntermediateBufferSizeImpl()
{
    return 0;
}

uint64_t Runner::GetTilingBufferSizeImpl()
{
    return 0;
}


Status Runner::FillHostTilingBufferImpl(uint8_t *hostTilingBuffer, uint64_t tilingBufferSize)
{
    ATB_LOG(INFO) << GetLogPrefix() << "hostTilingBuffer:" << hostTilingBuffer << ", tilingBufferSize"
                  << tilingBufferSize;
    return NO_ERROR;
}

uint64_t Runner::GetWorkspaceBufferSizeImpl()
{
    return 0;
}

Status Runner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack:" << runnerVariantPack.ToString();
    return NO_ERROR;
}

void Runner::SetSaveTensorDir(const std::string &tensorDir)
{
    tensorDir_ = tensorDir;
}

bool Runner::IsSaveTensor() const
{
    return Probe::IsExecuteCountInRange(executeCount_) && saveTensorFlag_;
}

std::string Runner::GetLogPrefix() const
{
    std::stringstream ss;
    ss << logPrefix_ << ":" << std::to_string(executeCount_) << " ";
    return ss.str();
}

std::string Runner::GetSaveTensorDir() const
{
    return tensorDir_;
}

void Runner::SetRunnerOperation(Operation *operation)
{
    operation_ = operation;
}

void Runner::DumpIOTensorInfo(RunnerVariantPack &runnerVariantPack) const
{
    OperationBase *opBase = dynamic_cast<OperationBase *>(operation_);
    if (!opBase) {
        ATB_LOG(ERROR) << GetLogPrefix() << "operation is not inherit from OperationBase";
        return;
    }

    size_t executeCount = executeCount_;
    const std::vector<int64_t> &operationBaseIds = opBase->GetOperationBaseIds();
    std::string opName = GenerateOperationName(operationName_, operationBaseIds);
    nlohmann::json opParamJson = opBase->GetParamJson();
    std::string opParam = opParamJson.dump();
    std::string tensorDir = tensorDir_ + "/" + (Probe::IsSaveTensorBefore() ? "before" : "after");
    std::vector<Probe::Tensor> inTensorsDesc(runnerVariantPack.inTensors.size());
    for (size_t i = 0; i < runnerVariantPack.inTensors.size(); i++) {
        std::string fileName = "intensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string tensorPath = Mki::FileSystem::Join({tensorDir, fileName});
        Probe::AtbTensorToProbeTensor(runnerVariantPack.inTensors[i], inTensorsDesc[i], tensorPath);
    }
    std::vector<Probe::Tensor> outTensorsDesc(runnerVariantPack.outTensors.size());
    for (size_t i = 0; i < runnerVariantPack.outTensors.size(); i++) {
        std::string fileName = "outtensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string tensorPath = Mki::FileSystem::Join({tensorDir, fileName});
        Probe::AtbTensorToProbeTensor(runnerVariantPack.outTensors[i], outTensorsDesc[i], tensorPath);
    }

    if (opParam == "null") {
        Probe::ReportOperationIOTensor(executeCount, opName, "{}", inTensorsDesc, outTensorsDesc);
    } else {
        Probe::ReportOperationIOTensor(executeCount, opName, opParam, inTensorsDesc, outTensorsDesc);
    }
}

void Runner::SetParam(const Mki::Any &param)
{
    (void)param;
}

bool Runner::IsSupportGlbWorkspace()
{
    return false;
}
} // namespace atb