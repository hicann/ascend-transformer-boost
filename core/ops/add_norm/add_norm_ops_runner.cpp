

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
#include "add_norm_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
AddNormOpsRunner::AddNormOpsRunner(const AddNormParam &param) : OpsRunner("AddNormOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "AddNormOperation::AddNormOperation called";
}

AddNormOpsRunner::~AddNormOpsRunner() {}

AsdOps::Status AddNormOpsRunner::Setup(VariantPack &variantPack)
{
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &yTensor = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors.at(3);
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);
    kernelGraph_.internalTensors.resize(3);
    AsdOps::Tensor &addNodeResultTensor = kernelGraph_.internalTensors.at(0);
    AsdOps::Tensor &layerNormMeanTensor = kernelGraph_.internalTensors.at(1);
    AsdOps::Tensor &layerNormVarianceTensor = kernelGraph_.internalTensors.at(2);

    kernelGraph_.nodes.resize(2);
    auto &addNode = kernelGraph_.nodes[0];
    auto &layerNormNode = kernelGraph_.nodes[1];

    addNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addNode.inTensors = {&xTensor, &yTensor};
    addNode.outTensors = {&addNodeResultTensor};

    int64_t beginDim = 0;
    if (!CalcLayerNormTensor(variantPack, beginDim)) {
        ASD_LOG(ERROR) << GetName() << " CalcLayerNormTensor fail";
        return AsdOps::Status::FailStatus(1, "CalcLayerNormTensor fail");
    }

    AsdOps::OpParam::Norm normParam = {AsdOps::OpParam::Norm::NORM_LAYERNORM};
    normParam.begin_norm_axis = beginDim;
    normParam.begin_params_axis = beginDim;
    normParam.epsilon = param_.layerNormEps;
    layerNormNode.opDesc = {0, "NormOperation", normParam};
    layerNormNode.inTensors = {&addNodeResultTensor, &weightTensor, &biasTensor};
    layerNormNode.outTensors = {&resultTensor, &layerNormMeanTensor, &layerNormVarianceTensor};

    return OpsRunner::Setup(variantPack);
}

// uint64_t AddNormOpsRunner::GetWorkspaceSize()
// {
//     const AsdOps::Tensor &layerNormMeanTensor = kernelGraph_.internalTensors.at(1);
//     const AsdOps::Tensor &layerNormVarianceTensor = kernelGraph_.internalTensors.at(2);
//     return layerNormMeanTensor.dataSize + layerNormVarianceTensor.dataSize + OpsRunner::GetWorkspaceSize();
// }

// AsdOps::Status AddNormOpsRunner::Execute(Handle &handle, VariantPack &variantPack)
// {
//     AsdOps::Tensor &layerNormMeanTensor = kernelGraph_.internalTensors.at(1);
//     AsdOps::Tensor &layerNormVarianceTensor = kernelGraph_.internalTensors.at(2);
//     layerNormMeanTensor.data = variantPack.workspace;
//     layerNormVarianceTensor.data = variantPack.workspace + layerNormMeanTensor.dataSize;

//     uint64_t ofset = layerNormMeanTensor.dataSize + layerNormVarianceTensor.dataSize;
//     VariantPack internalVarPack = variantPack;
//     internalVarPack.workspace += ofset;
//     internalVarPack.workspaceSize -= ofset;
//     return OpsRunner::Execute(handle, internalVarPack);
// }

bool AddNormOpsRunner::CalcLayerNormTensor(VariantPack &variantPack, int64_t &beginDim)
{
    AsdOps::TensorDesc inputDesc;
    inputDesc.dtype = variantPack.inTensors.at(0).desc.dtype;
    if (variantPack.inTensors.at(0).desc.dims.size() > variantPack.inTensors.at(1).desc.dims.size()) {
        inputDesc.dims = variantPack.inTensors.at(0).desc.dims;
    } else {
        inputDesc.dims = variantPack.inTensors.at(1).desc.dims;
    }

    AsdOps::Tensor &weightTensor = variantPack.inTensors.at(2);
    AsdOps::Tensor &biasTensor = variantPack.inTensors.at(3);

    ASD_LOG(INFO) << GetName() << " layer norm input desc:" << AsdOpsTensorDescToString(inputDesc)
                  << ", weightTensor:" << AsdOpsTensorToString(weightTensor)
                  << ", biasTensor:" << AsdOpsTensorToString(biasTensor);

    AsdOps::Tensor &layerNormMeanTensor = kernelGraph_.internalTensors.at(1);
    AsdOps::Tensor &layerNormVarianceTensor = kernelGraph_.internalTensors.at(2);

    const int axis = inputDesc.dims.size() - weightTensor.desc.dims.size();
    const int64_t M =
        std::accumulate(inputDesc.dims.begin(), inputDesc.dims.begin() + axis, 1LL, std::multiplies<int64_t>());
    const int64_t N =
        std::accumulate(inputDesc.dims.begin() + axis, inputDesc.dims.end(), 1LL, std::multiplies<int64_t>());

    ASD_LOG(INFO) << GetName() << " M:" << M;
    if (M < 0) {
        layerNormMeanTensor.desc.dtype = inputDesc.dtype;
        layerNormMeanTensor.desc.dims = {M};
        layerNormVarianceTensor.desc = layerNormMeanTensor.desc;
        return false;
    }

    int64_t numels = 1;
    AsdOps::SVector<int64_t> reduceDims; // the output of mean and rstd is Multidimension
    AsdOps::SVector<int64_t> weightDims; // the input of weight is Multidimension
    for (size_t i = 0; i < inputDesc.dims.size(); i++) {
        numels *= inputDesc.dims.at(i);
        reduceDims.emplace_back(inputDesc.dims.at(i));
        if (numels == M) {
            beginDim = i + 1;
            while (++i < inputDesc.dims.size()) {
                reduceDims.emplace_back(1);
                weightDims.emplace_back(inputDesc.dims.at(i));
            }
            break;
        }
    }
    layerNormMeanTensor.desc.dtype = weightTensor.desc.dtype;
    layerNormMeanTensor.desc.dims = reduceDims;
    layerNormVarianceTensor.desc = layerNormMeanTensor.desc;
    return true;
}
} // namespace AclTransformer
