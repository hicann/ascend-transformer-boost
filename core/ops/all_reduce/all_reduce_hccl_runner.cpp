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
#include "all_reduce_hccl_runner.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
AllReduceHcclRunner::AllReduceHcclRunner(const AllReduceParam &param)
    : HcclRunner("AllReduceHcclRunner", RUNNER_TYPE_ALL_REDUCE, param.rank, param.rankSize, param.rankRoot),
      param_(param)
{
    ASD_LOG(INFO) << "AllReduceHcclRunner::AllReduceHcclRunner called";
#ifdef USE_HCCL_RUNNER
    allReduceType_ = GetAllReduceType(param_.allReduceType);
#endif
}

#ifdef USE_HCCL_RUNNER
AllReduceHcclRunner::AllReduceHcclRunner(const AllReduceParam &param, HcclComm commExt)
    : HcclRunner("AllReduceHcclRunner", commExt, RUNNER_TYPE_ALL_REDUCE),
      param_(param)
{
    ASD_LOG(INFO) << "AllReduceHcclRunner::AllReduceHcclRunner Ext called";

    allReduceType_ = GetAllReduceType(param_.allReduceType);
}
#endif

AllReduceHcclRunner::~AllReduceHcclRunner() {}
} // namespace AclTransformer
