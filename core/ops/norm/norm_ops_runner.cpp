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
#include "norm_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
NormOpsRunner::NormOpsRunner(const NormParam &param) : OpsRunner("NormOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "NormOperation::NormOperation called";
}

NormOpsRunner::~NormOpsRunner() {}

AsdOps::Status NormOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    kernelGraph_.inTensors = variantPack.inTensors;
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors.at(2);
    kernelGraph_.outTensors = variantPack.outTensors;
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);
    kernelGraph_.internalTensors.resize(2);
    AsdOps::Tensor &layerNormMeanTensor = kernelGraph_.internalTensors.at(0);
    AsdOps::Tensor &layerNormVarianceTensor = kernelGraph_.internalTensors.at(1);

    kernelGraph_.nodes.resize(1);
    auto &layerNormNode = kernelGraph_.nodes[0];

    int64_t beginDim = 0;
    if (!CalcLayerNormTensor(variantPack, beginDim)) {
        ASD_LOG(ERROR) << GetName() << " CalcLayerNormTensor fail";
        return AsdOps::Status::FailStatus(1, "CalcLayerNormTensor fail");
    }

    AsdOps::OpParam::Norm normParam = {AsdOps::OpParam::Norm::NORM_LAYERNORM};
    normParam.begin_norm_axis = beginDim;
    normParam.begin_params_axis = beginDim;
    normParam.epsilon = param_.layerNormEps;
    ASD_LOG(INFO) << GetName() << " NormOperation opDesc normParam begin_norm_axis:" << normParam.begin_norm_axis
                  << ", begin_params_axis:" << normParam.begin_params_axis << ", epsilon:" << normParam.epsilon;
    layerNormNode.opDesc = {0, "NormOperation", normParam};
    layerNormNode.inTensors = {&xTensor, &weightTensor, &biasTensor};
    layerNormNode.outTensors = {&resultTensor, &layerNormMeanTensor, &layerNormVarianceTensor};
    return AsdOps::Status::OkStatus();
}

bool NormOpsRunner::CalcLayerNormTensor(const VariantPack &variantPack, int64_t &beginDim)
{
    const AsdOps::TensorDesc &inputDesc = variantPack.inTensors.at(0).desc;
    const AsdOps::Tensor &weightTensor = variantPack.inTensors.at(1);
    const AsdOps::Tensor &biasTensor = variantPack.inTensors.at(2);

    ASD_LOG(INFO) << GetName() << " layer norm input desc:" << TensorUtil::AsdOpsTensorDescToString(inputDesc)
                  << ", weightTensor:" << TensorUtil::AsdOpsTensorToString(weightTensor)
                  << ", biasTensor:" << TensorUtil::AsdOpsTensorToString(biasTensor);

    const int axis = inputDesc.dims.size() - weightTensor.desc.dims.size();
    const int64_t M =
        std::accumulate(inputDesc.dims.begin(), inputDesc.dims.begin() + axis, 1LL, std::multiplies<int64_t>());

    ASD_LOG(INFO) << GetName() << " M:" << M;
    if (M < 0) {
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

    return true;
}
} // namespace AclTransformer
