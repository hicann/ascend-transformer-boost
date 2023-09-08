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
#include "acltransformer/ops/multi_layer_linear_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "multi_layer_linear_ops_runner_builder.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
MultiLayerLinearOperation::MultiLayerLinearOperation(const MultiLayerLinearParam &param) : Operation("MultiLayerLinearOperation"), param_(param)
{
    runnerBuilders_ = {new MultiLayerLinearOpsRunnerBuilder(param_)};
}

MultiLayerLinearOperation::~MultiLayerLinearOperation() {}

uint64_t MultiLayerLinearOperation::GetInTensorCount() const { return 2; }

uint64_t MultiLayerLinearOperation::GetOutTensorCount() const { return 3; }

AsdOps::Status MultiLayerLinearOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1], inTensors.at(1).desc.dims[0] / 3};
    outTensorDescs.at(1) = inTensors.at(0).desc;
    outTensorDescs.at(1).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1], inTensors.at(1).desc.dims[0] / 3};
    outTensorDescs.at(2) = inTensors.at(0).desc;
    outTensorDescs.at(2).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1], inTensors.at(1).desc.dims[0] / 3};
    ASD_LOG(INFO) << "OutTensor dims = " << outTensorDescs.at(0).dims;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *MultiLayerLinearOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;

    return runnerBuilders_.at(index);
}
} // namespace AclTransformer