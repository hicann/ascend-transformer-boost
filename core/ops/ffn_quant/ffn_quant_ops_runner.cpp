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
#include "ffn_quant_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/matmul.h>

namespace AclTransformer {
FfnQuantOpsRunner::FfnQuantOpsRunner(const FfnQuantParam &param)
    : OpsRunner("FfnQuantOpsRunner", RUNNER_TYPE_FFN_QUANT), param_(param)
{
    ASD_LOG(INFO) << "FfnQuantOpsRunner::FfnQuantOpsRunner";
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &aTensor = kernelGraph_.inTensors[0];
    AsdOps::Tensor &bTensor = kernelGraph_.inTensors[1];
    AsdOps::Tensor &cTensor = kernelGraph_.inTensors[2];
    AsdOps::Tensor &dTensor = kernelGraph_.inTensors[3];
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[0];

    kernelGraph_.internalTensors.resize(1);
    AsdOps::Tensor &matmulOutTensor = kernelGraph_.internalTensors[0];

    kernelGraph_.nodes.resize(2);
    auto &matmulNode = kernelGraph_.nodes[0];
    auto &geluNode = kernelGraph_.nodes[1];
    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB})};
    matmulNode.inTensors = {&aTensor, &bTensor, &cTensor, &dTensor};
    matmulNode.outTensors = {&matmulOutTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    geluNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
    geluNode.inTensors = {&matmulOutTensor};
    geluNode.outTensors = {&operationOutTensor};
}

FfnQuantOpsRunner::~FfnQuantOpsRunner() {}
} // namespace AclTransformer