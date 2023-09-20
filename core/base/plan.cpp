/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "acltransformer/plan.h"
#include <unistd.h>
#include "acl/acl.h"
#include <cstring>
#include <syscall.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/status/status.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/runner.h"
#include "acltransformer/config.h"
#include "acltransformer/statistic.h"
#include "acltransformer/utils/profiling/profiling_funcs.h"

namespace AclTransformer {
const size_t HOST_TILING_BUFFER_DEFAULT_SIZE = 1024;

Plan::Plan()
{
    hostTilingBuffer_.resize(HOST_TILING_BUFFER_DEFAULT_SIZE);
    
    if (AsdOps::GetSingleton<Config>().IsUsingProfiling()) {
        hashIdArray_.resize(MAX_PROFILING_FUNC_NAME);
        const char *setUpName = "Plan::SetUp";
        const char *executeName = "Plan::Execute";
        const size_t setUpNameLen = strlen(setUpName);
        const size_t executeNameLen = strlen(executeName);
        hashIdArray_.at(PLAN_SETUP) = AsdOps::GetSingleton<AsdProfiling>().AsdGetHashId(setUpName, setUpNameLen);
        hashIdArray_.at(PLAN_EXECUTE) = AsdOps::GetSingleton<AsdProfiling>().AsdGetHashId(executeName, executeNameLen);
    }
}

Plan::~Plan() {}

void Plan::ReportApiInfo(const uint64_t beginTime, ProfilingFuncName type)
{
    MsProfApi info{};
    info.type = MSPROF_REPORT_ACL_OTHERS_BASE_TYPE;
    info.itemId = hashIdArray_.at(type);
    info.level = MSPROF_REPORT_ACL_LEVEL;
    const uint64_t endTime = AsdOps::GetSingleton<AsdProfiling>().AsdSysCycleTime();
    info.threadId = static_cast<uint32_t>(syscall(SYS_gettid));
    info.beginTime = beginTime;
    info.endTime = endTime;
    info.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    info.reserve = 0;
    auto ret = AsdOps::GetSingleton<AsdProfiling>().AsdReportApi(true, &info);
    if (ret != 0) {
        ASD_LOG(ERROR) << "AsdReportApi error!";
    }
}

void Plan::SetRunnerSaveTensorDir(const std::string &dir) { runner_->SetSaveTensorDir(dir + runner_->GetName()); }

AsdOps::Status Plan::Setup(Handle handle, const VariantPack &variantPack)
{
    const uint64_t beginTime = AsdOps::GetSingleton<Config>().IsUsingProfiling() 
                                ? AsdOps::GetSingleton<AsdProfiling>().AsdSysCycleTime()
                                : 0;
    ASD_LOG(INFO) << name_ << " setup start";
    runnerVariantPack_.inTensors = variantPack.inTensors;
    runnerVariantPack_.outTensors = variantPack.outTensors;
    runnerVariantPack_.param = variantPack.param;

    Reset();

    AsdOps::Status st = runner_->Setup(runnerVariantPack_);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " runner setup fail";
        return st;
    }

    runnerVariantPack_.tilingBufferSize = runner_->GetTilingBufferSize();
    if (runnerVariantPack_.tilingBufferSize > 0) {
        if (runnerVariantPack_.tilingBufferSize > hostTilingBuffer_.size()) {
            ASD_LOG(FATAL) << name_ << " resize host tiling buffer size from:" << hostTilingBuffer_.size()
                           << " to:" << runnerVariantPack_.tilingBufferSize;
            hostTilingBuffer_.resize(runnerVariantPack_.tilingBufferSize);
        }
        runner_->FillHostTilingBufferSize(hostTilingBuffer_.data(), runnerVariantPack_.tilingBufferSize);
    }

    runnerVariantPack_.workspaceBufferSize = runner_->GetWorkspaceBufferSize();
    runnerVariantPack_.intermediateBufferSize = runner_->GetIntermediateBufferSize();

    ASD_LOG(INFO) << name_ << " setup success, tilingBufferSize:" << runnerVariantPack_.tilingBufferSize
                  << ", workspaceBufferSize:" << runnerVariantPack_.workspaceBufferSize
                  << ", intermediateBufferSize:" << runnerVariantPack_.intermediateBufferSize;

    if (AsdOps::GetSingleton<Config>().IsUsingProfiling()) {
        ReportApiInfo(beginTime, PLAN_SETUP);
    }
    return AsdOps::Status::OkStatus();
}

uint64_t Plan::GetWorkspaceSize()
{
    return runnerVariantPack_.tilingBufferSize + runnerVariantPack_.workspaceBufferSize +
           runnerVariantPack_.intermediateBufferSize;
}

AsdOps::Status Plan::Execute(Handle handle, VariantPack &variantPack)
{
    const uint64_t beginTime = AsdOps::GetSingleton<Config>().IsUsingProfiling() 
                                    ? AsdOps::GetSingleton<AsdProfiling>().AsdSysCycleTime()
                                    : 0;
    ASD_LOG(INFO) << name_ << " execute start";
    runnerVariantPack_.inTensors = variantPack.inTensors;
    runnerVariantPack_.outTensors = variantPack.outTensors;
    runnerVariantPack_.param = variantPack.param;
    runnerVariantPack_.tilingBuffer = variantPack.workspace;
    runnerVariantPack_.workspaceBuffer = (uint8_t *)variantPack.workspace + runnerVariantPack_.tilingBufferSize;
    runnerVariantPack_.intermediateBuffer =
        (uint8_t *)variantPack.workspace + runnerVariantPack_.tilingBufferSize + runnerVariantPack_.workspaceBufferSize;

    AsdOps::Status st = CopyHostTilingToDevice(handle);
    if (!st.Ok()) {
        return st;
    }
#ifdef USE_TILING_STREAM
    aclrtRecordEvent(handle.copyEvent, handle.copyStream);
    aclrtStreamWaitEvent(handle.stream, handle.copyEvent);
    aclrtResetEvent(handle.copyEvent, handle.stream);
#endif

    st = runner_->Execute(handle, runnerVariantPack_);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " runner execute fail, error:" << st.Message();
        return st;
    }

    ASD_LOG(INFO) << name_ << " execute success";
    ASD_LOG(INFO) << name_ << " runner execute success";
    if (AsdOps::GetSingleton<Config>().IsUsingProfiling()) {
        ReportApiInfo(beginTime, PLAN_EXECUTE);
    }
    return AsdOps::Status::OkStatus();
}

void Plan::Reset()
{
    runnerVariantPack_.tilingBuffer = nullptr;
    runnerVariantPack_.tilingBufferSize = 0;
    runnerVariantPack_.workspaceBuffer = nullptr;
    runnerVariantPack_.workspaceBufferSize = 0;
    runnerVariantPack_.intermediateBuffer = nullptr;
    runnerVariantPack_.intermediateBufferSize = 0;
}

AsdOps::Status Plan::CopyHostTilingToDevice(Handle handle)
{
    if (runnerVariantPack_.tilingBufferSize > 0) {
        ASD_LOG(INFO) << name_ << " copy host tiling to device start, totalTilingBufferSize:"
                      << runnerVariantPack_.tilingBufferSize;
        AsdOps::Timer timer;
        void *stream = handle.stream;
#ifdef USE_TILING_STREAM
        stream = handle.copyStream;
#endif
        int ret = AsdRtMemCopyAsync(runnerVariantPack_.tilingBuffer, runnerVariantPack_.tilingBufferSize,
                                    hostTilingBuffer_.data(), runnerVariantPack_.tilingBufferSize,
                                    ASDRT_MEMCOPY_HOST_TO_DEVICE, stream);
        AsdOps::GetSingleton<Statistic>().tillingCopyTime += timer.ElapsedMicroSecond();
        if (ret != 0) {
            ASD_LOG(ERROR) << name_ << " copy host tiling to device fail, ret:" << ret;
            return AsdOps::Status::FailStatus(1, "copy host tiling to device fail");
        }
        ASD_LOG(INFO) << name_ << " copy host tiling to device success";
    }
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer