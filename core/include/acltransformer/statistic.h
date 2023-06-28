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
#ifndef ACLTRANSFORMER_STATISTIC_H
#define ACLTRANSFORMER_STATISTIC_H
#include <string>

namespace AclTransformer {
struct Statistic {
    uint64_t totalTime = 0;
    uint64_t syclTime = 0;
    uint64_t tillingCopyTime = 0;
    uint64_t getBestKernelTime = 0;
    uint64_t planSetupTime = 0;
    uint64_t planExecuteTime = 0;
    uint64_t kernelCacheHitCount_ = 0;
    uint64_t kernelCacheMissCount_ = 0;

    std::string ToString() const;
    void Reset();
};
} // namespace AclTransformer
#endif