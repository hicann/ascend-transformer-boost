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
#include "acltransformer/ops/multinomial_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "multinomial_ops_runner_builder.h"
#include <asdops/utils/log/log.h>
namespace AclTransformer {
MultinomialOperation::MultinomialOperation(const MultinomialParam &param)
    : Operation("MultinomialOperation"), param_(param)
{
    runnerBuilders_ = {new MultinomialOpsRunnerBuilder(param)};
}

MultinomialOperation::~MultinomialOperation() {}

uint64_t MultinomialOperation::GetInTensorCount() const { return 1; }

uint64_t MultinomialOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status MultinomialOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                    AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims[0] = inTensors.at(0).desc.dims[0];
    outTensorDescs.at(0).dims[1] = param_.numSamples;
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    outTensorDescs.at(0).dtype = AsdOps::TensorDType::TENSOR_DTYPE_UINT32;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *MultinomialOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer
