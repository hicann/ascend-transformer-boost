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
#include <asdops/utils/log/log.h>
#include <asdops/utils/status/status.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/runner.h"
#include "acltransformer/statistic.h"

namespace AclTransformer {
const size_t HOST_TILING_BUFFER_DEFAULT_SIZE = 1024;

Plan::Plan() { hostTilingBuffer_.resize(HOST_TILING_BUFFER_DEFAULT_SIZE); }

Plan::~Plan() {}

AsdOps::Status Plan::Setup(Handle handle, const VariantPack &variantPack)
{
    runnerVariantPack_.inTensors = variantPack.inTensors;
    runnerVariantPack_.outTensors = variantPack.outTensors;

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

    ASD_LOG(INFO) << name_ << " runner setup success, tilingBufferSize:" << runnerVariantPack_.tilingBufferSize
                  << ", workspaceBufferSize:" << runnerVariantPack_.workspaceBufferSize
                  << ", intermediateBufferSize:" << runnerVariantPack_.intermediateBufferSize;
    return AsdOps::Status::OkStatus();
}

uint64_t Plan::GetWorkspaceSize()
{
    return runnerVariantPack_.tilingBufferSize + runnerVariantPack_.workspaceBufferSize +
           runnerVariantPack_.intermediateBufferSize;
}

AsdOps::Status Plan::Execute(Handle handle, VariantPack &variantPack)
{
    runnerVariantPack_.inTensors = variantPack.inTensors;
    runnerVariantPack_.outTensors = variantPack.outTensors;
    runnerVariantPack_.tilingBuffer = variantPack.workspace;
    runnerVariantPack_.workspaceBuffer = (char *)variantPack.workspace + runnerVariantPack_.tilingBufferSize;
    runnerVariantPack_.intermediateBuffer =
        (char *)variantPack.workspace + runnerVariantPack_.tilingBufferSize + runnerVariantPack_.workspaceBufferSize;

    AsdOps::Status st = CopyHostTilingToDevice(handle);
    if (!st.Ok()) {
        return st;
    }

    st = runner_->Execute(handle, runnerVariantPack_);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " runner execute fail, error:" << st.Message();
        return st;
    }

    ASD_LOG(INFO) << name_ << " runner execute success";
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
        int ret = AsdRtMemCopyAsync(runnerVariantPack_.tilingBuffer, runnerVariantPack_.tilingBufferSize,
                                    hostTilingBuffer_.data(), runnerVariantPack_.tilingBufferSize,
                                    ASDRT_MEMCOPY_HOST_TO_DEVICE, handle.stream);
        AsdOps::GetSingleton<Statistic>().tillingCopyTime = timer.ElapsedMicroSecond();
        if (ret != 0) {
            ASD_LOG(ERROR) << name_ << " copy host tiling to device fail, ret:" << ret;
            return AsdOps::Status::FailStatus(1, "copy host tiling to device fail");
        }
    }
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer