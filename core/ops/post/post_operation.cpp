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
#include "acltransformer/ops/post_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "post_torch_runner_builder.h"

namespace AclTransformer {
PostOperation::PostOperation(const PostParam &param) : Operation("PostOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new PostTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new PostOpsRunnerBuilder(param_)};
#endif
}

PostOperation::~PostOperation() {}

uint64_t PostOperation::GetInTensorCount() const { return 1; }

uint64_t PostOperation::GetOutTensorCount() const { return 2; }

AsdOps::Status PostOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                             AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype =  AsdOps::TensorDType::TENSOR_DTYPE_INT64;
    outTensorDescs.at(0).dims = {1, param_.min_tokens_to_keep};
    outTensorDescs.at(0).format = AsdOps::TensorFormat::TENSOR_FORMAT_ND;

    outTensorDescs.at(1).dtype =  AsdOps::TensorDType::TENSOR_DTYPE_INT64;
    outTensorDescs.at(1).dims = {1, param_.top_k};
    outTensorDescs.at(1).format = AsdOps::TensorFormat::TENSOR_FORMAT_ND;

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *PostOperation::FindBestRunnerBuilder() const
{
#ifdef USE_TORCH_RUNNER
    size_t index = 0; // ops_runner has not been developed
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer