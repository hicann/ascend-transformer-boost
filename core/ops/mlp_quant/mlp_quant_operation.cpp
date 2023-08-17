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
#include "acltransformer/ops/mlp_quant_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
//#include "mlp_torch_runner_builder.h"
#include "mlp_quant_ops_runner_builder.h"

static constexpr int64_t DEFAULT_IN_TENSOR_SIZE = 10;
static constexpr int64_t DEFAULT_OUT_TENSOR_SIZE = 1;

namespace AclTransformer {
MlpQuantOperation::MlpQuantOperation(const MlpQuantParam &param) : Operation("MlpQuantOperation"), param_(param)
{
    runnerBuilders_ = {new MlpQuantOpsRunnerBuilder(param_)};
}

MlpQuantOperation::~MlpQuantOperation() {}

uint64_t MlpQuantOperation::GetInTensorCount() const
{
    return DEFAULT_IN_TENSOR_SIZE;
}

uint64_t MlpQuantOperation::GetOutTensorCount() const { return DEFAULT_OUT_TENSOR_SIZE; }

AsdOps::Status MlpQuantOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dtype = AsdOps::TENSOR_DTYPE_FLOAT16;
    ASD_LOG(INFO) << "infer shape, model name is " << param_.model;

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *MlpQuantOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer