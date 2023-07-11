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
#include "acltransformer/ops/add_norm_quant_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "add_norm_quant_ops_runner_builder.h"

namespace AclTransformer {
AddNormQuantOperation::AddNormQuantOperation(const AddNormQuantParam &param)
    : Operation("AddNormQuantOperation"), param_(param)
{
    runnerBuilders_ = {new AddNormQuantOpsRunnerBuilder(param_)};
}

AddNormQuantOperation::~AddNormQuantOperation() {}

uint64_t AddNormQuantOperation::GetInTensorCount() const { 
	return 4; 
	}

uint64_t AddNormQuantOperation::GetOutTensorCount() const { 
	return 2; 
	}

AsdOps::Status AddNormQuantOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                     AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = inTensors.at(0).desc;
    outTensorDescs.at(0).dtype = AsdOps::TENSOR_DTYPE_INT8;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *AddNormQuantOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer