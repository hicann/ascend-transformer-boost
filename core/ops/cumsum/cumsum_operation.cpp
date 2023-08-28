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
#include "acltransformer/ops/cumsum_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "cumsum_ops_runner_builder.h"
#include "acltransformer/params/cumsum.h"

namespace AclTransformer {
CumsumOperation::CumsumOperation(const CumsumParam &param) : Operation("CumsumOperation"), param_(param)
{
    runnerBuilders_ = {new CumsumOpsRunnerBuilder(param_)};
}

CumsumOperation::~CumsumOperation() {}

uint64_t CumsumOperation::GetInTensorCount() const { return 1; }

uint64_t CumsumOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status CumsumOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                               AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    outTensorDescs.at(0).dims = inTensors.at(0).desc.dims;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *CumsumOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer
