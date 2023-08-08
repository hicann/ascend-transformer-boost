/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include "test_common.h"

at::IntArrayRef ToIntArrayRef(const AsdOps::SVector<int64_t> &src) { return at::IntArrayRef(src.data(), src.size()); }

at::IntArrayRef ToIntArrayRef(const std::vector<int64_t> &src) { return at::IntArrayRef(src.data(), src.size()); }

int64_t Prod(const AsdOps::SVector<int64_t> &vec)
{
    int64_t ret = 1;
    for (auto x : vec) {
        ret *= x;
    }
    return ret;
}