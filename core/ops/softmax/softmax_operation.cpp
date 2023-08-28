/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 *  * Licensed under the Apache License, Version 2.0 (the "License");
 *  * you may not use this file except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 *  * http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  */
#include "acltransformer/ops/softmax_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "softmax_ops_runner_builder.h"
#include <asdops/utils/log/log.h>
namespace AclTransformer {
SoftmaxOperation::SoftmaxOperation(const SoftmaxParam &param) : Operation("SoftmaxOperation"), param_(param)
{
    runnerBuilders_ = {new SoftmaxOpsRunnerBuilder(param)};
}

SoftmaxOperation::~SoftmaxOperation() {}

uint64_t SoftmaxOperation::GetInTensorCount() const { return 1; }

uint64_t SoftmaxOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status SoftmaxOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SoftmaxOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer
