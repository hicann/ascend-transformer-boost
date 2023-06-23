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
#include "acltransformer/statistic.h"

namespace AclTransformer {
std::string Statistic::ToString() const
{
    return "layerExecTime:" + std::to_string(layerExecTime) + ", " + "syclTime:" + std::to_string(syclTime) + ", " +
           "tillingCopyTime:" + std::to_string(tillingCopyTime) +
           ", getBestKernelTime:" + std::to_string(getBestKernelTime);
}

void Statistic::Reset()
{
    layerExecTime = 0;
    syclTime = 0;
    tillingCopyTime = 0;
    getBestKernelTime = 0;
}
} // namespace AclTransformer