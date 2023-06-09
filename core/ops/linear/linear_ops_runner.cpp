

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
#include "linear_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
LinearOpsRunner::LinearOpsRunner(LinearParam &param) : OpsRunner("LinearOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "LinearOperation::LinearOperation called";
}

LinearOpsRunner::~LinearOpsRunner() {}

AsdOps::Status LinearOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    if (variantPack.inTensors.at(1).desc.format == AsdOps::TENSOR_FORMAT_FRACTAL_NZ) {
        ASD_LOG(INFO) << GetName() << " SetupKernelGraph b format is nz";
        AsdOps::SVector<int64_t> matmulOrgShape;
        AsdOps::SVector<int64_t> transdataOrgShape;
        VariantPack newVariantPack;
        ConvertNewVariantPack(variantPack, newVariantPack, matmulOrgShape, transdataOrgShape);
        auto &bTensor = newVariantPack.inTensors.at(1);
        int64_t bLastDim = bTensor.desc.dims.at(bTensor.desc.dims.size() - 1);
        if (bLastDim != 16 || bTensor.desc.dims.size() < 4) {
            AsdOps::SVector<int64_t> bTensorNewDims = {1, bLastDim / 16, bTensor.desc.dims.at(1), 16};
            bTensor.View(bTensorNewDims);
            ASD_LOG(INFO) << GetName()
                          << " bTensor last dim is not 16, view:" << TensorUtil::AsdOpsTensorDescToString(bTensor.desc);
        }

        ASD_LOG(INFO) << GetName() << " Setup variantPack:" << variantPack.ToString()
                      << ", newVariantPack:" << newVariantPack.ToString();

        kernelGraph_.inTensors = newVariantPack.inTensors;
        AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[0];
        AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[1];
        AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[2];

        kernelGraph_.outTensors = newVariantPack.outTensors;
        AsdOps::Tensor &resultTensor = kernelGraph_.outTensors[0];
        kernelGraph_.internalTensors.resize(3);
        AsdOps::Tensor &transdata0ResultTensor = kernelGraph_.internalTensors[0];
        AsdOps::Tensor &matmulResultTensor = kernelGraph_.internalTensors[1];
        AsdOps::Tensor &transdata2ResultTensor = kernelGraph_.internalTensors[2];

        kernelGraph_.nodes.resize(4);
        auto &transdata0Node = kernelGraph_.nodes[0];
        auto &matmulNode = kernelGraph_.nodes[1];
        auto &transdata2Node = kernelGraph_.nodes[2];
        auto &addNode = kernelGraph_.nodes[3];

        transdata0Node.opDesc = {0, "TransdataOperation",
                                 AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
        transdata0Node.inTensors = {&inputTensor};
        transdata0Node.outTensors = {&transdata0ResultTensor};

        ASD_LOG(INFO) << GetName() << " MatMulOperation orgShape:[" << matmulOrgShape.at(0) << ", "
                      << matmulOrgShape.at(1) << ", " << matmulOrgShape.at(2) << "]";
        matmulNode.opDesc = {0, "MatMulOperation",
                             AsdOps::OpParam::MatMul({param_.transposeA, param_.transposeB, matmulOrgShape})};
        matmulNode.inTensors = {&transdata0ResultTensor, &weightTensor};
        matmulNode.outTensors = {&matmulResultTensor};

        ASD_LOG(INFO) << GetName() << " Transdata orgShape:[" << transdataOrgShape.at(0) << ", "
                      << transdataOrgShape.at(1) << "]";
        transdata2Node.opDesc = {
            0, "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transdataOrgShape})};
        transdata2Node.inTensors = {&matmulResultTensor};
        transdata2Node.outTensors = {&transdata2ResultTensor};

        addNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
        addNode.inTensors = {&transdata2ResultTensor, &biasTensor};
        addNode.outTensors = {&resultTensor};
    } else {
        AsdOps::SVector<int64_t> matmulOrgShape;
        AsdOps::SVector<int64_t> transdataOrgShape;
        VariantPack newVariantPack;
        ConvertNewVariantPack(variantPack, newVariantPack, matmulOrgShape, transdataOrgShape);
        ASD_LOG(INFO) << GetName() << " Setup variantPack:" << variantPack.ToString()
                      << ", newVariantPack:" << newVariantPack.ToString();

        kernelGraph_.inTensors = newVariantPack.inTensors;
        AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[0];
        AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[1];
        AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[2];

        kernelGraph_.outTensors = newVariantPack.outTensors;
        AsdOps::Tensor &resultTensor = kernelGraph_.outTensors[0];
        kernelGraph_.internalTensors.resize(4);
        AsdOps::Tensor &transdata0ResultTensor = kernelGraph_.internalTensors[0];
        AsdOps::Tensor &transdata1ResultTensor = kernelGraph_.internalTensors[1];
        AsdOps::Tensor &matmulResultTensor = kernelGraph_.internalTensors[2];
        AsdOps::Tensor &transdata2ResultTensor = kernelGraph_.internalTensors[3];

        kernelGraph_.nodes.resize(5);
        auto &transdata0Node = kernelGraph_.nodes[0];
        auto &transdata1Node = kernelGraph_.nodes[1];
        auto &matmulNode = kernelGraph_.nodes[2];
        auto &transdata2Node = kernelGraph_.nodes[3];
        auto &addNode = kernelGraph_.nodes[4];

        transdata0Node.opDesc = {0, "TransdataOperation",
                                 AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
        transdata0Node.inTensors = {&inputTensor};
        transdata0Node.outTensors = {&transdata0ResultTensor};

        transdata1Node.opDesc = {0, "TransdataOperation",
                                 AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
        transdata1Node.inTensors = {&weightTensor};
        transdata1Node.outTensors = {&transdata1ResultTensor};

        ASD_LOG(INFO) << GetName() << " MatMulOperation orgShape:[" << matmulOrgShape.at(0) << ", "
                      << matmulOrgShape.at(1) << ", " << matmulOrgShape.at(2) << "]";
        matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true, matmulOrgShape})};
        matmulNode.inTensors = {&transdata0ResultTensor, &transdata1ResultTensor};
        matmulNode.outTensors = {&matmulResultTensor};

        ASD_LOG(INFO) << GetName() << " Transdata orgShape:[" << transdataOrgShape.at(0) << ", "
                      << transdataOrgShape.at(1) << "]";
        transdata2Node.opDesc = {
            0, "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transdataOrgShape})};
        transdata2Node.inTensors = {&matmulResultTensor};
        transdata2Node.outTensors = {&transdata2ResultTensor};

        addNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
        addNode.inTensors = {&transdata2ResultTensor, &biasTensor};
        addNode.outTensors = {&resultTensor};
    }

    return AsdOps::Status::OkStatus();
}

AsdOps::Status LinearOpsRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    VariantPack newVariantPack;
    AsdOps::SVector<int64_t> matmulOrgShape;
    AsdOps::SVector<int64_t> transdataOrgShape;
    ConvertNewVariantPack(variantPack, newVariantPack, matmulOrgShape, transdataOrgShape);
    return OpsRunner::ExecuteImpl(handle, newVariantPack);
}

void LinearOpsRunner::ConvertNewVariantPack(const VariantPack &variantPack, VariantPack &newVariantPack,
                                            AsdOps::SVector<int64_t> &matmulOrgShape,
                                            AsdOps::SVector<int64_t> &transdataOrgShape)
{
    newVariantPack = variantPack;
    AsdOps::Tensor &aTensor = newVariantPack.inTensors.at(0);
    AsdOps::Tensor &bTensor = newVariantPack.inTensors.at(1);
    if (aTensor.desc.dims.size() == 1 || aTensor.desc.dims.size() == 2) {
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
} // namespace AclTransformer
