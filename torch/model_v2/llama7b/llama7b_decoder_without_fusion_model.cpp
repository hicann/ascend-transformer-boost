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
#include "models/llama13b/llama13blayer_param_parallel.h"
#include "models/llama13b/llama13blayer_parallel_operation.h"
#include "torch/model_v2/llama7b/llama7b_decoder_without_fusion_model.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 9;
const int INPUT_TENSOR_COUNT_BEFORE_KEY = 5;
const int OUTPUT_TENSOR_COUNT_BEFORE_KEY = 1;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY
};

void Llama7BDecoderWithoutFusionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    ASD_LOG(INFO) << "ChatGlm2DecoderModel param rmsNormEps:" << rmsNormEps
                  << ", headNum:" << headNum  
                  << ", dk:" << dk
                  << ", layerNum:" << layerNum
                  << ", rank:" << rank
                  << ", rankSize:" << rankSize;
}

Llama7BDecoderWithoutFusionModel::Llama7BDecoderWithoutFusionModel(const std::string &param) : Model("ChatGlm2DecoderModel", param)
{
    param_.FromString(param);
}

Llama7BDecoderWithoutFusionModel::~Llama7BDecoderWithoutFusionModel() {}

uint64_t Llama7BDecoderWithoutFusionModel::GetInTensorCount() const
{
    return graph_.inTensors.size(); 
}

uint64_t Llama7BDecoderWithoutFusionModel::GetOutTensorCount() const
{
    return graph_.outTensors.size();
}

AsdOps::Status Llama7BDecoderWithoutFusionModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    ASD_LOG(INFO) << "Enter Llama7BDecoderWithoutFusionModel InferShape";
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

void Llama7BDecoderWithoutFusionModel::BuildGraph()
{
    ASD_LOG(INFO) << "Enter Llama7BDecoderWithoutFusionModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(INPUT_TENSOR_COUNT_BEFORE_KEY + param_.layerNum * 2);
    graph_.outTensors.resize(OUTPUT_TENSOR_COUNT_BEFORE_KEY + param_.layerNum * 2);

    const int nodeSize = param_.layerNum;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(0);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        LLaMA13BLayerParam modelParam;
        modelParam.rmsNormEps = param_.rmsNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.model = "llama13b";
        modelParam.rank = param_.rank;
        modelParam.rankSize = param_.rankSize;

        layerNode.operation = std::make_shared<LLaMA13BLayerOperation>(modelParam);
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
}

AsdOps::Status Llama7BDecoderWithoutFusionModel::ParseVarintPackParam(
    const std::string &param, int nodeId, AsdOps::Any &variantPackParam)
{
    return AsdOps::Status::OkStatus();
}

}
