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
AsdProfiling::AsdProfiling() {}

AsdProfiling::~AsdProfiling() {}

int32_t AsdProfiling::AsdReportApi(uint32_t agingFlag, const MsProfApi *api) { return MsprofReportApi(agingFlag, api); }

int32_t AsdProfiling::AsdReportCompactInfo(uint32_t agingFlag, void *data, uint32_t length)
{
    return MsprofReportCompactInfo(agingFlag, data, length);
}

uint64_t AsdProfiling::AsdSysCycleTime() { return MsprofSysCycleTime(); }

uint64_t AsdProfiling::AsdGetHashId(const char *hashInfo, size_t length) { return MsprofGetHashId(hashInfo, length); }
} // namespace AclTransformer
