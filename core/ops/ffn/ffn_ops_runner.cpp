/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "ffn_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/matmul.h>

namespace AclTransformer {
FfnOpsRunner::FfnOpsRunner(const FfnParam &param) : OpsRunner("FfnOpsRunner"), param_(param) {}

AsdOps::Status FfnOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    kernelGraph_.inTensors = variantPack.inTensors;
    AsdOps::Tensor &aTensor = kernelGraph_.inTensors[0];
    AsdOps::Tensor &bTensor = kernelGraph_.inTensors[1];
    AsdOps::Tensor &cTensor = kernelGraph_.inTensors[2];
    kernelGraph_.outTensors = variantPack.outTensors;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[0];

    kernelGraph_.internalTensors.resize(5);
    AsdOps::Tensor &castAOutTensor = kernelGraph_.internalTensors[0];
    AsdOps::Tensor &castBOutTensor = kernelGraph_.internalTensors[1];
    AsdOps::Tensor &matmulOutTensor = kernelGraph_.internalTensors[2];
    AsdOps::Tensor &castMatmulOutTensor = kernelGraph_.internalTensors[3];
    AsdOps::Tensor &addOutTensor = kernelGraph_.internalTensors[4];

    kernelGraph_.nodes.resize(6);
    auto &castANode = kernelGraph_.nodes[0];
    auto &castBNode = kernelGraph_.nodes[1];
    auto &matmulNode = kernelGraph_.nodes[2];
    auto &castMatmulNode = kernelGraph_.nodes[3];
    auto &addNode = kernelGraph_.nodes[4];
    auto &geluNode = kernelGraph_.nodes[5];

    castANode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castANode.inTensors = {&aTensor};
    castANode.outTensors = {&castAOutTensor};
    castBNode.opDesc = castANode.opDesc;
    castBNode.inTensors = {&bTensor};
    castBNode.outTensors = {&castBOutTensor};

    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul()};
    matmulNode.inTensors = {&castAOutTensor, &castBOutTensor};
    matmulNode.outTensors = {&matmulOutTensor};

    castMatmulNode.opDesc = castANode.opDesc;
    castMatmulNode.inTensors = {&matmulOutTensor};
    castMatmulNode.outTensors = {&castMatmulOutTensor};

    addNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addNode.inTensors = {&castMatmulOutTensor, &cTensor};
    addNode.outTensors = {&addOutTensor};

    geluNode.opDesc = {0, "ActivationOperation",
                       AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_GELU})};
    geluNode.inTensors = {&addOutTensor};
    geluNode.outTensors = {&operationOutTensor};

    return AsdOps::Status::OkStatus();
}

FfnOpsRunner::~FfnOpsRunner() {}
} // namespace AclTransformer