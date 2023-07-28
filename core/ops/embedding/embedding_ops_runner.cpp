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
#include "embedding_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
EmbeddingOpsRunner::EmbeddingOpsRunner(const EmbeddingParam &param) : OpsRunner("EmbeddingOpsRunner", RUNNER_TYPE_EMBEDDING), param_(param)
{
    ASD_LOG(INFO) << "EmbeddingOpsRunner::EmbeddingOpsRunner called";

    kernelGraph_.nodes.resize(1);
    int64_t nodeNum = 0;
    auto &embeddingNode = kernelGraph_.nodes[nodeNum];

    const int DIME_TWO = 2;
    kernelGraph_.inTensors.resize(DIME_TWO);
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &tableTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &indicsTensor = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &embeddedTensor = kernelGraph_.outTensors.at(0);
    embeddingNode.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {param.axis}}};
    embeddingNode.inTensors = {&tableTensor, &indicsTensor};
    embeddingNode.outTensors = {&embeddedTensor};
}

EmbeddingOpsRunner::~EmbeddingOpsRunner() {}
} // namespace AclTransformer
