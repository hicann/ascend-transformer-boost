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
#include "acltransformer/ops/all_reduce_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/config.h"
#include "all_reduce_lccl_runner_builder.h"

namespace AclTransformer {
AllReduceOperation::AllReduceOperation(const AllReduceParam &param) : Operation("AllReduceOperation"), param_(param)
{
    ASD_LOG(INFO) << "AllReduceOperation::AllReduceOperation called";
    runnerBuilders_ = {new AllReduceLcclRunnerBuilder(param_)};
}

AllReduceOperation::~AllReduceOperation() {}

uint64_t AllReduceOperation::GetInTensorCount() const { return 1; }

uint64_t AllReduceOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status AllReduceOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *AllReduceOperation::FindBestRunnerBuilder() const
{
    return runnerBuilders_.at(0);
}
} // namespace AclTransformer