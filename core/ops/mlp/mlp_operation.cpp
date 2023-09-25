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
#include "acltransformer/ops/mlp_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "mlp_torch_runner_builder.h"
#include "mlp_ops_runner_builder.h"

static constexpr int64_t GLM2_6B_IN_TENSOR_SIZE = 3;
static constexpr int64_t GLM2_6B_IN_TENSOR_PARALLEL_SIZE = 2;
static constexpr int64_t GLM130B_IN_TENSOR_SIZE = 3;
static constexpr int64_t LLAMA13B_IN_TENSOR_SIZE = 3;
static constexpr int64_t LLAMA_ADAPTER_IN_TENSOR_SIZE = 7;
static constexpr int64_t DEFAULT_IN_TENSOR_SIZE = 4;
namespace AclTransformer {
MlpOperation::MlpOperation(const MlpParam &param) : Operation("MlpOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new MlpOpsRunnerBuilder(param_), new MlpTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new MlpOpsRunnerBuilder(param_)};
#endif
}

MlpOperation::~MlpOperation() {}

uint64_t MlpOperation::GetInTensorCount() const
{
    if (param_.model == "glm130b") {
        return GLM130B_IN_TENSOR_SIZE;
    } else if (param_.model == "chatglm2_6b") {
        return GLM2_6B_IN_TENSOR_SIZE;
    } else if (param_.model == "chatglm2_6b_parallel") {
        return GLM2_6B_IN_TENSOR_PARALLEL_SIZE;
    } else if (param_.model == "llama13b" || param_.model == "llama65b" || param_.model == "llama70b") {
        return LLAMA13B_IN_TENSOR_SIZE;
    } else if (param_.model == "llama_adapter") {
        return LLAMA_ADAPTER_IN_TENSOR_SIZE;
    } else {
        return DEFAULT_IN_TENSOR_SIZE;
    }
}

uint64_t MlpOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status MlpOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    ASD_LOG(INFO) << "infer shape, model name is " << param_.model;
    if (param_.model == "glm130b") {
        auto outTensorDim0 = inTensors.at(0).desc.dims[0];
        auto outTensorDim1 = inTensors.at(0).desc.dims[1];
        auto outTensorDim2 = inTensors.at(1).desc.dims[0] / 2;
        outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1, outTensorDim2};
    } else if (param_.model == "chatglm2_6b") {
        auto outTensorDim0 = inTensors.at(0).desc.dims[0];
        auto outTensorDim1 = inTensors.at(0).desc.dims[1];
        auto outTensorDim2 = inTensors.at(0).desc.dims[2];
        outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1, outTensorDim2};
    } else if (param_.model == "chatglm2_6b_parallel") {
        auto outTensorDim0 = inTensors.at(0).desc.dims[0];
        auto outTensorDim1 = inTensors.at(0).desc.dims[1];
        auto outTensorDim2 = inTensors.at(1).desc.dims[0] / 2;
        outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1, outTensorDim2};
    } else if (param_.model == "llama13b" || param_.model == "llama65b" || param_.model == "llama70b") {
        auto outTensorDim0 = inTensors.at(0).desc.dims[0];
        auto outTensorDim1 = inTensors.at(0).desc.dims[1];
        auto outTensorDim2 = inTensors.at(1).desc.dims[0];
        outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1, outTensorDim2};
    } else if (param_.model == "llama_adapter") {
        auto outTensorDim0 = inTensors.at(0).desc.dims[0];
        auto outTensorDim1 = inTensors.at(0).desc.dims[1];
        auto outTensorDim2 = inTensors.at(0).desc.dims[2];
        outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1, outTensorDim2};
    }

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *MlpOperation::FindBestRunnerBuilder() const
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsMlpOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer