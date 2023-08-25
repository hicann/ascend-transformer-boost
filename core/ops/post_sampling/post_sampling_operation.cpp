/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, s
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "acltransformer/ops/post_sampling_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "post_sampling_ops_runner_builder.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
PostSamplingOperation::PostSamplingOperation(const PostSamplingParam &param) : Operation("PostSamplingOperation"), param_(param)
{
    runnerBuilders_ = {new PostSamplingOpsRunnerBuilder(param_)};
}

PostSamplingOperation::~PostSamplingOperation() {}

uint64_t PostSamplingOperation::GetInTensorCount() const { return 3; }

uint64_t PostSamplingOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status PostSamplingOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0]};
    ASD_LOG(INFO) << "OutTensor dims = " << outTensorDescs.at(0).dims;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *PostSamplingOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;

    return runnerBuilders_.at(index);
}
} // namespace AclTransformer