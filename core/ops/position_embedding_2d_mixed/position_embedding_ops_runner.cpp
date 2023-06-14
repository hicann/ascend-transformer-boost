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
#include "position_embedding_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingOpsRunner::PositionEmbeddingOpsRunner(const PositionEmbeddingParam &param)
    : OpsRunner("PositionEmbeddingOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingOperation::PositionEmbeddingOperation called";
}

PositionEmbeddingOpsRunner::~PositionEmbeddingOpsRunner() {}

AsdOps::Status PositionEmbeddingOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " SetupKernelGraph start: " << "headNum: " << param_.headNum;

    kernelGraph_.inTensors = variantPack.inTensors;
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors = variantPack.outTensors;
    AsdOps::Tensor &qEmbed = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbed = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(8);
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(0);

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
