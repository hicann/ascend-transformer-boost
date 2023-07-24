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

#include "acltransformer/utils/profiling/profiling_funcs.h"
#include <dlfcn.h>
#include <asdops/utils/log/log.h>

namespace AclTransformer {
AsdProfiling::AsdProfiling() { kernelNameHashCache_.resize(RUNNER_TYPE_MAX); }

AsdProfiling::~AsdProfiling() {}

void AsdProfiling::Init(RunnerType runnerType, uint64_t kernelCount)
{
    if (runnerType == RUNNER_TYPE_UNDEFINED) {
        return;
    }
    if (kernelNameHashCache_.at(runnerType).empty()) {
        kernelNameHashCache_.at(runnerType).resize(kernelCount);
    }
}

int32_t AsdProfiling::AsdReportApi(uint32_t agingFlag, const MsProfApi *api)
{
    ASD_LOG(INFO) << "AsdReportApi start!";
    return MsprofReportApi(agingFlag, api);
}

int32_t AsdProfiling::AsdReportCompactInfo(uint32_t agingFlag, void *data, uint32_t length)
{
    ASD_LOG(INFO) << "AsdReportCompactInfo start!";
    return MsprofReportCompactInfo(agingFlag, data, length);
}

uint64_t AsdProfiling::AsdSysCycleTime() { return MsprofSysCycleTime(); }

uint64_t AsdProfiling::AsdGetHashId(const char *hashInfo, size_t length) { return MsprofGetHashId(hashInfo, length); }

uint64_t AsdProfiling::AsdGetHashId(const char *hashInfo, size_t length, RunnerType runnerType, size_t nodeId)
{
    if (runnerType >= 0 && runnerType < RUNNER_TYPE_MAX) {
        if (nodeId >= 0 && (uint64_t)nodeId < kernelNameHashCache_.at(runnerType).size()) {
            auto hashId = kernelNameHashCache_.at(runnerType).at(nodeId);
            if (hashId != 0) {
                return hashId;
            }
        }
    }

    auto hashId = MsprofGetHashId(hashInfo, length);

    if (runnerType >= 0 && runnerType < RUNNER_TYPE_MAX) {
        if (nodeId >= 0 && (uint64_t)nodeId < kernelNameHashCache_.at(runnerType).size()) {
            kernelNameHashCache_.at(runnerType).at(nodeId) = hashId;
        }
    }
    return hashId;
}
} // namespace AclTransformer
