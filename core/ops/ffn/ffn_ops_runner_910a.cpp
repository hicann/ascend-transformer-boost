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
#include "ffn_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/matmul.h>

namespace AclTransformer {
FfnOpsRunner910A::FfnOpsRunner910A(const FfnParam &param) : OpsRunner("FfnOpsRunner910A", RUNNER_TYPE_FFN), param_(param)
{
    ASD_LOG(INFO) << "FfnOpsRunner910A::FfnOpsRunner910A";
}

FfnOpsRunner910A::~FfnOpsRunner910A() {}

AsdOps::Status FfnOpsRunner910A::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    RunnerVariantPack newRunnerVariantPack;
    AsdOps::SVector<int64_t> matmulOrgShape;
    AsdOps::SVector<int64_t> transdataOrgShape;
    ConvertNewRunnerVariantPackA(runnerVariantPack, newRunnerVariantPack, matmulOrgShape, transdataOrgShape);
    return OpsRunner::ExecuteImpl(handle, newRunnerVariantPack);
}

void FfnOpsRunner910A::ConvertNewRunnerVariantPackA(const RunnerVariantPack &runnerVariantPack,
                                                       RunnerVariantPack &newRunnerVariantPack,
                                                       AsdOps::SVector<int64_t> &matmulOrgShape,
                                                       AsdOps::SVector<int64_t> &transdataOrgShape)
{
    const std::size_t dim2 = 2;
    newRunnerVariantPack = runnerVariantPack;
    AsdOps::Tensor &aTensor = newRunnerVariantPack.inTensors.at(0);
    AsdOps::Tensor &bTensor = newRunnerVariantPack.inTensors.at(1);
    if (aTensor.desc.dims.size() == 1 || aTensor.desc.dims.size() == dim2) {
        return;
    }

    int64_t dimZero = 1;
    for (size_t i = 0; i < aTensor.desc.dims.size() - 1; ++i) {
        dimZero *= aTensor.desc.dims.at(i);
    }
    AsdOps::SVector<int64_t> newDims = {dimZero, aTensor.desc.dims.at(aTensor.desc.dims.size() - 1)};

    ASD_LOG(INFO) << GetName() << " old aTensor:" << TensorUtil::AsdOpsTensorToString(aTensor)
                  << ", bTensor:" << TensorUtil::AsdOpsTensorToString(bTensor);

    aTensor.View(newDims);
    ASD_LOG(INFO) << GetName() << " after view, new aTensor:" << TensorUtil::AsdOpsTensorToString(aTensor);

    if (param_.transposeB) {
        matmulOrgShape = {aTensor.desc.dims.at(0), aTensor.desc.dims.at(1), bTensor.desc.dims.at(0)};
        transdataOrgShape = {aTensor.desc.dims.at(0), bTensor.desc.dims.at(0)};
    } else {
        matmulOrgShape = {aTensor.desc.dims.at(0), aTensor.desc.dims.at(1), bTensor.desc.dims.at(1)};
        transdataOrgShape = {aTensor.desc.dims.at(0), bTensor.desc.dims.at(1)};
    }

    aTensor.AddDimOne();
    ASD_LOG(INFO) << GetName() << " after add one, new aTensor:" << TensorUtil::AsdOpsTensorToString(aTensor);

    bTensor.AddDimOne();
    ASD_LOG(INFO) << GetName() << " after add one, new bTensor:" << TensorUtil::AsdOpsTensorToString(bTensor);
}

AsdOps::Status FfnOpsRunner910A::SetupKernelGraphNz(const RunnerVariantPack &runnerVariantPack)
{
    const std::size_t nodeSize = 5;
    const std::size_t internalTensorsSize = 4;
    const int64_t blockDim = 16;
    const std::size_t dimSize = 4;
    const std::size_t id0 = 0;
    const std::size_t id1 = 1;
    const std::size_t id2 = 2;
    const std::size_t id3 = 3;
    const std::size_t id4 = 4;
    ASD_LOG(INFO) << GetName() << " SetupKernelGraph b format is nz";
    AsdOps::SVector<int64_t> matmulOrgShape;
    AsdOps::SVector<int64_t> transdataOrgShape;
    RunnerVariantPack newRunnerVariantPack;
    ConvertNewRunnerVariantPackA(runnerVariantPack, newRunnerVariantPack, matmulOrgShape, transdataOrgShape);
    auto &bTensor = newRunnerVariantPack.inTensors.at(1);
    int64_t bLastDim = bTensor.desc.dims.at(bTensor.desc.dims.size() - 1);
    if (bLastDim != blockDim || bTensor.desc.dims.size() < dimSize) {
        AsdOps::SVector<int64_t> bTensorNewDims = {1, bLastDim / blockDim, bTensor.desc.dims.at(1), blockDim};
        bTensor.View(bTensorNewDims);
        ASD_LOG(INFO) << GetName()
                      << " bTensor last dim is not 16, view:" << TensorUtil::AsdOpsTensorDescToString(bTensor.desc);
    }

    ASD_LOG(INFO) << GetName() << " Setup runnerVariantPack:" << runnerVariantPack.ToString()
                  << ", newRunnerVariantPack:" << newRunnerVariantPack.ToString();

    kernelGraph_.inTensors = newRunnerVariantPack.inTensors;
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[id0];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[id1];
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[id2];

    kernelGraph_.outTensors = newRunnerVariantPack.outTensors;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[id0];
    kernelGraph_.internalTensors.resize(internalTensorsSize);
    AsdOps::Tensor &transdata0ResultTensor = kernelGraph_.internalTensors[id0];
    AsdOps::Tensor &matmulResultTensor = kernelGraph_.internalTensors[id1];
    AsdOps::Tensor &transdata2ResultTensor = kernelGraph_.internalTensors[id2];
    AsdOps::Tensor &resultTensor = kernelGraph_.internalTensors[id3];

    kernelGraph_.nodes.resize(nodeSize);
    auto &transdata0Node = kernelGraph_.nodes[id0];
    auto &matmulNode = kernelGraph_.nodes[id1];
    auto &transdata2Node = kernelGraph_.nodes[id2];
    auto &addNode = kernelGraph_.nodes[id3];
    auto &geluNode = kernelGraph_.nodes[id4];

    transdata0Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata0Node.inTensors = {&inputTensor};
    transdata0Node.outTensors = {&transdata0ResultTensor};

    ASD_LOG(INFO) << GetName() << " MatMulOperation orgShape:[" << TensorUtil::AsdOpsDimsToString(matmulOrgShape)
                  << "]";
    matmulNode.opDesc = {0, "MatMulOperation",
                         AsdOps::OpParam::MatMul({param_.transposeA, param_.transposeB, matmulOrgShape})};
    matmulNode.inTensors = {&transdata0ResultTensor, &weightTensor};
    matmulNode.outTensors = {&matmulResultTensor};

    ASD_LOG(INFO) << GetName() << " Transdata orgShape:[" << TensorUtil::AsdOpsDimsToString(transdataOrgShape) << "]";
    transdata2Node.opDesc = {
        0, "TransdataOperation",
        AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transdataOrgShape})};
    transdata2Node.inTensors = {&matmulResultTensor};
    transdata2Node.outTensors = {&transdata2ResultTensor};

    addNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addNode.inTensors = {&transdata2ResultTensor, &biasTensor};
    addNode.outTensors = {&resultTensor};

    geluNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
    geluNode.inTensors = {&resultTensor};
    geluNode.outTensors = {&operationOutTensor};

    return AsdOps::Status::OkStatus();
}

AsdOps::Status FfnOpsRunner910A::SetupKernelGraphNd(const RunnerVariantPack &runnerVariantPack)
{
    const std::size_t nodeSize = 6;
    const std::size_t internalTensorsSize = 5;
    const std::size_t id0 = 0;
    const std::size_t id1 = 1;
    const std::size_t id2 = 2;
    const std::size_t id3 = 3;
    const std::size_t id4 = 4;
    const std::size_t id5 = 5;
    AsdOps::SVector<int64_t> matmulOrgShape;
    AsdOps::SVector<int64_t> transdataOrgShape;
    RunnerVariantPack newRunnerVariantPack;
    ConvertNewRunnerVariantPackA(runnerVariantPack, newRunnerVariantPack, matmulOrgShape, transdataOrgShape);
    ASD_LOG(INFO) << GetName() << " Setup runnerVariantPack:" << runnerVariantPack.ToString()
                  << ", newRunnerVariantPack:" << newRunnerVariantPack.ToString();

    kernelGraph_.inTensors = newRunnerVariantPack.inTensors;
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[id0];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[id1];
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[id2];

    kernelGraph_.outTensors = newRunnerVariantPack.outTensors;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[id0];
    kernelGraph_.internalTensors.resize(internalTensorsSize);
    AsdOps::Tensor &transdata0ResultTensor = kernelGraph_.internalTensors[id0];
    AsdOps::Tensor &transdata1ResultTensor = kernelGraph_.internalTensors[id1];
    AsdOps::Tensor &matmulResultTensor = kernelGraph_.internalTensors[id2];
    AsdOps::Tensor &transdata2ResultTensor = kernelGraph_.internalTensors[id3];
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors[id4];

    kernelGraph_.nodes.resize(nodeSize);
    auto &transdata0Node = kernelGraph_.nodes[id0];
    auto &transdata1Node = kernelGraph_.nodes[id1];
    auto &matmulNode = kernelGraph_.nodes[id2];
    auto &transdata2Node = kernelGraph_.nodes[id3];
    auto &addNode = kernelGraph_.nodes[id4];
    auto &geluNode = kernelGraph_.nodes[id5];

    transdata0Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata0Node.inTensors = {&inputTensor};
    transdata0Node.outTensors = {&transdata0ResultTensor};

    transdata1Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata1Node.inTensors = {&weightTensor};
    transdata1Node.outTensors = {&transdata1ResultTensor};

    ASD_LOG(INFO) << GetName() << " MatMulOperation orgShape:[" << TensorUtil::AsdOpsDimsToString(matmulOrgShape)
                  << "]";
    matmulNode.opDesc = {0, "MatMulOperation",
                         AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB, matmulOrgShape})};
    matmulNode.inTensors = {&transdata0ResultTensor, &transdata1ResultTensor};
    matmulNode.outTensors = {&matmulResultTensor};

    ASD_LOG(INFO) << GetName() << " Transdata orgShape:[" << TensorUtil::AsdOpsDimsToString(transdataOrgShape) << "]";
    transdata2Node.opDesc = {
        0, "TransdataOperation",
        AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transdataOrgShape})};
    transdata2Node.inTensors = {&matmulResultTensor};
    transdata2Node.outTensors = {&transdata2ResultTensor};

    addNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addNode.inTensors = {&transdata2ResultTensor, &biasTensor};
    addNode.outTensors = {&resultTensor};

    geluNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
    geluNode.inTensors = {&resultTensor};
    geluNode.outTensors = {&operationOutTensor};
    return AsdOps::Status::OkStatus();
}

AsdOps::Status FfnOpsRunner910A::SetupKernelGraph(const RunnerVariantPack &runnerVariantPack)
{
    if (runnerVariantPack.inTensors.at(1).desc.format == AsdOps::TENSOR_FORMAT_FRACTAL_NZ) {
        return SetupKernelGraphNz(runnerVariantPack);
    } else {
        return SetupKernelGraphNd(runnerVariantPack);
    }
}
} // namespace AclTransformer