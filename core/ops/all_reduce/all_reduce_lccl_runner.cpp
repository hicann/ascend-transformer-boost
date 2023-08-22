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
#include "all_reduce_lccl_runner.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
AllReduceLcclRunner::AllReduceLcclRunner(const AllReduceParam &param) : 
    LcclRunner("AllReduceLcclRunner", RUNNER_TYPE_ALL_REDUCE, param.rank, param.rankSize), param_(param)
{
    ASD_LOG(INFO) << "AllReduceLcclRunner::AllReduceLcclRunner called";
#ifdef USE_LCCL_RUNNER
    allReduceType_ = GetAllReduceType(param_.allReduceType);
#endif
}

AllReduceLcclRunner::~AllReduceLcclRunner() {}
} // namespace AclTransformer
