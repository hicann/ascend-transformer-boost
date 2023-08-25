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
#include "ffn_quant_runner_310p.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/matmul.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
FfnQuantOpsRunner310P::FfnQuantOpsRunner310P(const FfnQuantParam &param)
    : OpsRunner("FfnQuantOpsRunner310P", RUNNER_TYPE_FFN_QUANT), param_(param)
{
    ASD_LOG(INFO) << "FfnQuantOpsRunner310P::FfnQuantOpsRunner310P";
}

FfnQuantOpsRunner310P::~FfnQuantOpsRunner310P() {}

AsdOps::Status FfnQuantOpsRunner310P::SetupKernelGraphNz(const RunnerVariantPack &runnerVariantPack)
{
    const std::size_t nodeSize = 4;
    const std::size_t internalTensorsSize = 4;
    const std::size_t dimSize = 4;

    ASD_LOG(INFO) << GetName() << " SetupKernelGraph b format is nz";

    kernelGraph_.inTensors = runnerVariantPack.inTensors;
    int64_t inTensorNum = 0;
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &scaleTensor = kernelGraph_.inTensors[inTensorNum++];

    kernelGraph_.outTensors = runnerVariantPack.outTensors;
    int64_t outTensorNum = 0;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[outTensorNum++];

    kernelGraph_.internalTensors.resize(internalTensorsSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &transdata0ResultTensor = kernelGraph_.internalTensors[internalTensorNum++];
    AsdOps::Tensor &matmulResultTensor = kernelGraph_.internalTensors[internalTensorNum++];
    AsdOps::Tensor &transdata2ResultTensor = kernelGraph_.internalTensors[internalTensorNum++];
    AsdOps::Tensor &resultTensor = kernelGraph_.internalTensors[internalTensorNum++];

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &transdata0Node = kernelGraph_.nodes[nodeNum++];
    auto &matmulNode = kernelGraph_.nodes[nodeNum++];
    auto &transdata2Node = kernelGraph_.nodes[nodeNum++];
    auto &activateNode = kernelGraph_.nodes[nodeNum++];

    ViewFunc Squeeze1 = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimA_ = oldDims;
        if (oldDims.size() == 2) {
            oriSize_ = 2;
            newDims = {1, oldDims.at(0), oldDims.at(1)};
        } else {
            newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
        }
    };

    transdata0Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata0Node.inTensors = {&inputTensor};
    transdata0Node.outTensors = {&transdata0ResultTensor};
    transdata0Node.inTensorViewFuncs.resize(transdata0Node.inTensors.size());
    transdata0Node.inTensorViewFuncs.at(0) = Squeeze1;

    ASD_LOG(INFO) << GetName() << " MatMulOperation orgShape:[" << TensorUtil::AsdOpsDimsToString({0, 0}) << "]";
    ViewFunc CheckDimB = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        int64_t bLastDim = oldDims.at(oldDims.size() - 1);
        const int64_t blockDim = 16;
        if (bLastDim != blockDim || oldDims.size() < dimSize) {
            newDims = {1, bLastDim / blockDim, oldDims.at(0), blockDim};
        }
        oriDimB_ = {newDims.at(2), newDims.at(1) * newDims.at(3)};
    };
    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({param_.transposeA, param_.transposeB, {0, 0}})};
    matmulNode.inTensors = {&transdata0ResultTensor, &weightTensor, &biasTensor,&scaleTensor};
    matmulNode.outTensors = {&matmulResultTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(1) = CheckDimB;
    matmulNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1, dim2;
        if (oriSize_ == 3) {
            dim0 = oriDimA_.at(0) * oriDimA_.at(1);
            dim1 = oriDimA_.at(2);
        } else {
            dim0 = oriDimA_.at(0);
            dim1 = oriDimA_.at(1);
        }
        if (!param_.transposeB) {
            dim2 = oriDimB_.at(0);
        } else {
            dim2 = oriDimB_.at(1);
        }
        runInfo.SetOpDesc({0, "MatMulOperation",
                           AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB, {dim0, dim1, dim2}})});
    };

    ASD_LOG(INFO) << GetName() << " Transdata orgShape:[" << TensorUtil::AsdOpsDimsToString({0, 0}) << "]";
    transdata2Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata2Node.inTensors = {&matmulResultTensor};
    transdata2Node.outTensors = {&transdata2ResultTensor};
    transdata2Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1;
        if (oriSize_ == 3) {
            dim0 = oriDimA_.at(0) * oriDimA_.at(1);
        } else {
            dim0 = oriDimA_.at(0);
        }
        if (!param_.transposeB) {
            dim1 = oriDimB_.at(0);
        } else {
            dim1 = oriDimB_.at(1);
        }
        runInfo.SetOpDesc({0, "TransdataOperation",
                           AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {dim0, dim1}})});
    };
    switch (param_.activationFuncType) {
    case FfnQuantParam::ActivationFuncType::FAST_GELU:
        activateNode.opDesc = {0, "ElewiseOperation",
                               AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
        break;
    case FfnQuantParam::ActivationFuncType::GELU:
        activateNode.opDesc = {0, "ActivationOperation",
                               AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_GELU})};
        break;
    case FfnQuantParam::ActivationFuncType::RELU:
        activateNode.opDesc = {0, "ActivationOperation",
                               AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_RELU})};
        break;
    default:
        activateNode.opDesc = {0, "ElewiseOperation",
                               AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
        break;
    }

    activateNode.inTensors = {&transdata2ResultTensor};
    activateNode.outTensors = {&operationOutTensor};

    return AsdOps::Status::OkStatus();
}

AsdOps::Status FfnQuantOpsRunner310P::SetupKernelGraphNd(const RunnerVariantPack &runnerVariantPack)
{
    const std::size_t nodeSize = 5;
    const std::size_t internalTensorsSize = 4;

    kernelGraph_.inTensors = runnerVariantPack.inTensors;
    int64_t inTensorNum = 0;
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &scaleTensor = kernelGraph_.inTensors[inTensorNum++];

    kernelGraph_.outTensors = runnerVariantPack.outTensors;
    int64_t outTensorNum = 0;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[outTensorNum++];

    kernelGraph_.internalTensors.resize(internalTensorsSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &transdata0ResultTensor = kernelGraph_.internalTensors[internalTensorNum++];
    AsdOps::Tensor &transdata1ResultTensor = kernelGraph_.internalTensors[internalTensorNum++];
    AsdOps::Tensor &matmulResultTensor = kernelGraph_.internalTensors[internalTensorNum++];
    AsdOps::Tensor &transdata2ResultTensor = kernelGraph_.internalTensors[internalTensorNum++];

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &transdata0Node = kernelGraph_.nodes[nodeNum++];
    auto &transdata1Node = kernelGraph_.nodes[nodeNum++];
    auto &matmulNode = kernelGraph_.nodes[nodeNum++];
    auto &transdata2Node = kernelGraph_.nodes[nodeNum++];
    auto &activateNode = kernelGraph_.nodes[nodeNum++];

    ViewFunc Squeeze1 = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimA_ = oldDims;
        if (oldDims.size() == 2) {
            oriSize_ = 2;
            newDims = {1, oldDims.at(0), oldDims.at(1)};
        } else {
            newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
        }
    };

    transdata0Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata0Node.inTensors = {&inputTensor};
    transdata0Node.outTensors = {&transdata0ResultTensor};
    transdata0Node.inTensorViewFuncs.resize(transdata0Node.inTensors.size());
    transdata0Node.inTensorViewFuncs.at(0) = Squeeze1;

    ViewFunc CheckDimB = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimB_ = oldDims;
        newDims = {1, oldDims.at(0), oldDims.at(1)};
    };
    transdata1Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata1Node.inTensors = {&weightTensor};
    transdata1Node.outTensors = {&transdata1ResultTensor};
    transdata1Node.inTensorViewFuncs.resize(transdata1Node.inTensors.size());
    transdata1Node.inTensorViewFuncs.at(0) = CheckDimB;

    ASD_LOG(INFO) << GetName() << " MatMulOperation orgShape:[" << TensorUtil::AsdOpsDimsToString({0, 0}) << "]";
    matmulNode.opDesc = {0, "MatMulOperation",
                         AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB, {0, 0}})};
    matmulNode.inTensors = {&transdata0ResultTensor, &transdata1ResultTensor,&biasTensor,&scaleTensor};
    matmulNode.outTensors = {&matmulResultTensor};
    matmulNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1, dim2;
        if (oriSize_ == 3) {
            dim0 = oriDimA_.at(0) * oriDimA_.at(1);
            dim1 = oriDimA_.at(2);
        } else {
            dim0 = oriDimA_.at(0);
            dim1 = oriDimA_.at(1);
        }
        if (!param_.transposeB) {
            dim2 = oriDimB_.at(0);
        } else {
            dim2 = oriDimB_.at(1);
        }
        ASD_LOG(FATAL) << dim0 << " " << dim1 << " " << dim2;
        runInfo.SetOpDesc({0, "MatMulOperation",
                           AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB, {dim0, dim1, dim2}})});
    };

    ASD_LOG(INFO) << GetName() << " Transdata orgShape:[" << TensorUtil::AsdOpsDimsToString({0, 0}) << "]";
    transdata2Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata2Node.inTensors = {&matmulResultTensor};
    transdata2Node.outTensors = {&transdata2ResultTensor};
    transdata2Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1;
        if (oriSize_ == 3) {
            dim0 = oriDimA_.at(0) * oriDimA_.at(1);
        } else {
            dim0 = oriDimA_.at(0);
        }
        if (!param_.transposeB) {
            dim1 = oriDimB_.at(0);
        } else {
            dim1 = oriDimB_.at(1);
        }
        runInfo.SetOpDesc({0, "TransdataOperation",
                           AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {dim0, dim1}})});
    };

    switch (param_.activationFuncType) {
    case FfnQuantParam::ActivationFuncType::FAST_GELU:
        activateNode.opDesc = {0, "ElewiseOperation",
                               AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
        break;
    case FfnQuantParam::ActivationFuncType::GELU:
        activateNode.opDesc = {0, "ActivationOperation",
                               AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_GELU})};
        break;
    case FfnQuantParam::ActivationFuncType::RELU:
        activateNode.opDesc = {0, "ActivationOperation",
                               AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_RELU})};
        break;
    default:
        activateNode.opDesc = {0, "ElewiseOperation",
                               AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
        break;
    }

    activateNode.inTensors = {&transdata2ResultTensor};
    activateNode.outTensors = {&operationOutTensor};

    return AsdOps::Status::OkStatus();
}

AsdOps::Status FfnQuantOpsRunner310P::SetupKernelGraph(const RunnerVariantPack &runnerVariantPack)
{
    if (runnerVariantPack.inTensors.at(1).desc.format == AsdOps::TENSOR_FORMAT_FRACTAL_NZ){
        return SetupKernelGraphNz(runnerVariantPack);  
    }else{
        return SetupKernelGraphNd(runnerVariantPack);
    } 
}
} // namespace AclTransformer