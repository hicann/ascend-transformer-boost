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
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/rms_norm_operation.h"
#include "models/llama70b/llama70blayer_parallel_operation.h"
#include "models/llama70b/llama70blayer_param_parallel.h"
#include "torch/model_v2/llama70b/llama70b_decoder_model.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 9;
const int INPUT_TENSOR_COUNT_BEFORE_KEY = 5;
const int OUTPUT_TENSOR_COUNT_BEFORE_KEY = 1;
const int RMSNORMNODE_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_AFTER_LAYER = 1;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY
};

void Llama70BDecoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    kvHeadNum = paramJson["kvHeadNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    ASD_LOG(INFO) << "Llama70BDecoderModel param rmsNormEps:" << rmsNormEps
                  << ", headNum:" << headNum
                  << ", kvHeadNum:" << kvHeadNum
                  << ", dk:" << dk
                  << ", layerNum:" << layerNum
                  << ", rank:" << rank
                  << ", rankSize:" << rankSize;
}

Llama70BDecoderModel::Llama70BDecoderModel(const std::string &param) : Model("Llama70BDecoderModel", param)
{
    param_.FromString(param);
}

Llama70BDecoderModel::~Llama70BDecoderModel() {}

uint64_t Llama70BDecoderModel::GetInTensorCount() const
{
    return graph_.inTensors.size(); 
}

uint64_t Llama70BDecoderModel::GetOutTensorCount() const
{
    return graph_.outTensors.size();
}

AsdOps::Status Llama70BDecoderModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    ASD_LOG(INFO) << "Enter Llama70BDecoderModel InferShape";
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    const AsdOps::Tensor &hiddenState = inTensors.at(IN_TENSOR_HIDDENSTATES);
    const AsdOps::Tensor &keyTensor = inTensors.at(IN_TENSOR_PAST_KEY);
    const AsdOps::Tensor &valueTensor = inTensors.at(IN_TENSOR_PAST_KEY + param_.layerNum);

    outTensorDescs.at(0) = hiddenState.desc;
    for (size_t keyId = 0; keyId < param_.layerNum; ++keyId) {
        outTensorDescs.at(1 + keyId) = keyTensor.desc;
        outTensorDescs.at(1 + keyId).dims.at(0) += 1;   
    }
    for (size_t valueId = 0; valueId < param_.layerNum; ++valueId) {
        outTensorDescs.at(1 + param_.layerNum + valueId) = valueTensor.desc;
        outTensorDescs.at(1 + param_.layerNum + valueId).dims.at(0) += 1;   
    }

    return AsdOps::Status::OkStatus();
}

void Llama70BDecoderModel::BuildGraph()
{
    ASD_LOG(INFO) << "Enter Llama70BDecoderModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum + RMSNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(INPUT_TENSOR_COUNT_BEFORE_KEY + param_.layerNum * 2);
    graph_.outTensors.resize(OUTPUT_TENSOR_COUNT_BEFORE_KEY + param_.layerNum * 2);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(0);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        LLaMA70BLayerParam modelParam;
        modelParam.rmsNormEps = param_.rmsNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.model = "llama13b";
        modelParam.rank = param_.rank;
        modelParam.rankSize = param_.rankSize;

        layerNode.operation = std::make_shared<LLaMA70BLayerOperation>(modelParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY + layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY + layerId + param_.layerNum);

        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(layerId),
                                    &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0),
                                    &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        }

        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &rmsNormNode = graph_.nodes.at(nodeId++);
    RmsNormParam rmsNormParam = {param_.rmsNormEps};
    rmsNormNode.operation = std::make_shared<RmsNormOperation>(rmsNormParam);
    const int rmsNormWeightTensorId = graph_.weightTensors.size() - 1;
    rmsNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(rmsNormWeightTensorId)};
    rmsNormNode.outTensors = {&graph_.outTensors.at(0)};
}

AsdOps::Status Llama70BDecoderModel::ParseVarintPackParam(
    const std::string &param, int nodeId, AsdOps::Any &variantPackParam)
{
    return AsdOps::Status::OkStatus();
}

}
