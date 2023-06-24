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
#include "acltransformer/ops/any_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "any_ops_runner_builder.h"

namespace AclTransformer {
AnyOperation::AnyOperation(const AnyParam &param) : Operation("AnyOperation"), param_(param)
{
    runnerBuilders_ = {new AnyOpsRunnerBuilder(param_)};
}

AnyOperation::~AnyOperation() {}

AsdOps::Status AnyOperation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                        AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensors.size() != 2) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 2");
    }

    outTensorDescs.resize(1);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *AnyOperation::FindBestRunnerBuilder() { return runnerBuilders_.at(0); }
} // namespace AclTransformer