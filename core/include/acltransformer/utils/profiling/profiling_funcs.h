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
#ifndef PROFILING_FUNCS_H
#define PROFILING_FUNCS_H

#include <vector>
#include "prof_api.h"
#include "acltransformer/runner_type.h"

namespace AclTransformer {
class AsdProfiling {
public:
    AsdProfiling();
    ~AsdProfiling();
    int32_t AsdReportApi(uint32_t agingFlag, const MsProfApi *api);
    int32_t AsdReportCompactInfo(uint32_t agingFlag, void *data, uint32_t length);
    uint64_t AsdSysCycleTime();
    uint64_t AsdGetHashId(const char *hashInfo, size_t length);
    uint64_t AsdGetHashId(const char *hashInfo, size_t length, RunnerType runnerType, size_t nodeId);
    void Init(RunnerType runnerType, uint64_t kernelCount);

private:
    std::vector<std::vector<uint64_t>> kernelNameHashCache_;
};
} // namespace AclTransformer
#endif