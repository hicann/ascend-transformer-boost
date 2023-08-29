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
#include "all_gather_hccl_runner.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
AllGatherHcclRunner::AllGatherHcclRunner(const AllGatherParam &param)
    : HcclRunner("AllGatherHcclRunner", RUNNER_TYPE_ALL_GATHER, param.rank, param.rankSize, param.rankRoot),
      param_(param)
{
    ASD_LOG(INFO) << "AllGatherHcclRunner::AllGatherHcclRunner called";
}

#ifdef USE_HCCL_RUNNER
AllGatherHcclRunner::AllGatherHcclRunner(const AllGatherParam &param, void *commExt)
    : HcclRunner("AllGatherHcclRunner", commExt, RUNNER_TYPE_ALL_GATHER),
      param_(param)
{
    ASD_LOG(INFO) << "AllGatherHcclRunner::AllGatherHcclRunner Ext called";
}
#endif

AllGatherHcclRunner::~AllGatherHcclRunner() {}
} // namespace AclTransformer
