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
#ifndef ALL_GATHER_HCCL_RUNNER_H
#define ALL_GATHER_HCCL_RUNNER_H
#include "acltransformer/base/hccl_runner.h"
#include "acltransformer/params/all_gather.h"

namespace AclTransformer {
class AllGatherHcclRunner : public HcclRunner {
public:
    AllGatherHcclRunner(const AllGatherParam &param);
#ifdef USE_HCCL_RUNNER
    AllGatherHcclRunner(const AllGatherParam &param, void *commExt);
#endif
    virtual ~AllGatherHcclRunner();

private:
    AllGatherParam param_;
};

} // namespace AclTransformer
#endif