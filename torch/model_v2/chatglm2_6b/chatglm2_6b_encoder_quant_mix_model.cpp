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
#include "chatglm2_6b_encoder_quant_mix_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "models/chatglm2_6b/chatglm2_6b_quant_layer_encoder_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"

#include "chatglm2_6b_encoder_model.h"
#include "models/chatglm2_6b/chatglm2_6b_layer_encoder_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 14;
const int WEIGHT_COUNT_FIRST_LAYER = 7;
const int FINALNORMNODE_WEIGHT_COUNT = 1;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_ROPECACHE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BETA,
    IN_TENSOR_MAX,
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void ChatGlm2QuantMixEncoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    transKey = paramJson["transKey"].get<bool>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int>();
    numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int>();
    qkvInputScale = paramJson["qkvInputScale"].get<std::vector<float>>();
    qkvInputOffset = paramJson["qkvInputOffset"].get<std::vector<int>>();
    denseInputScale = paramJson["denseInputScale"].get<std::vector<float>>();
    denseInputOffset = paramJson["denseInputOffset"].get<std::vector<int>>();
    selfLnInputScale = paramJson["selfLnInputScale"].get<std::vector<float>>();
    selfLnInputOffset = paramJson["selfLnInputOffset"].get<std::vector<int>>();
    ffnOutInputScale = paramJson["ffnOutInputScale"].get<std::vector<float>>();
    ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<std::vector<int>>();

    ASD_LOG(INFO) << "ChatGlm2QuantMixEncoderModel param rmsNormEps:" << rmsNormEps 
                  << ", transKey:" << transKey <<  ", layerNum:" << layerNum
                  << ", residualAddScale:" << residualAddScale  << ", numHeadsPerPartition: " << numHeadsPerPartition
                  << ", hiddenSizePerHead:" << hiddenSizePerHead << ", numGroupsPerPartition:" << numGroupsPerPartition;
}

ChatGlm2QuantMixEncoderModel::ChatGlm2QuantMixEncoderModel(const std::string &param) : Model("ChatGlm2QuantMixEncoderModel", param)
{
    param_.FromString(param);
}

ChatGlm2QuantMixEncoderModel::~ChatGlm2QuantMixEncoderModel() {}

uint64_t ChatGlm2QuantMixEncoderModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t ChatGlm2QuantMixEncoderModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status ChatGlm2QuantMixEncoderModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = inTensors.at(IN_TENSOR_HIDDENSTATES).desc;
    
    outTensorDescs.at(1) = inTensors.at(IN_TENSOR_HIDDENSTATES).desc;
    outTensorDescs.at(1).dims.clear();
    outTensorDescs.at(1).dims.push_back(inTensors.at(IN_TENSOR_HIDDENSTATES).desc.dims.at(0));
    outTensorDescs.at(1).dims.push_back(inTensors.at(IN_TENSOR_HIDDENSTATES).desc.dims.at(1));
    outTensorDescs.at(1).dims.push_back(param_.numGroupsPerPartition);
    outTensorDescs.at(1).dims.push_back(param_.hiddenSizePerHead);

    for (size_t i = 2; i < outTensorDescs.size(); ++i) {
        outTensorDescs.at(i) = outTensorDescs.at(1);
    }
    return AsdOps::Status::OkStatus();
}

void ChatGlm2QuantMixEncoderModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_FIRST_LAYER + WEIGHT_COUNT_PER_LAYER * (param_.layerNum-1) + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    // param_.layerNum * 2 (one for pastK, one for pastV)
    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX + param_.layerNum * 2);
    graph_.nodes.resize(param_.layerNum + FINALNORMNODE_WEIGHT_COUNT);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(IN_TENSOR_HIDDENSTATES);

    int noQuantNodeId = 0;
    int firstLayerId = 0;
    auto &firstLayerNode = graph_.nodes.at(noQuantNodeId++);
    ChatGlm2LayerParam opParamFirst;
    opParamFirst.numHeadsPerPartition = param_.numHeadsPerPartition;
    opParamFirst.numGroupsPerPartition = param_.numGroupsPerPartition;
    opParamFirst.hiddenSizePerHead = param_.hiddenSizePerHead;
    opParamFirst.layerId = firstLayerId;
    opParamFirst.rmsNormEps = param_.rmsNormEps;
    opParamFirst.residualAddScale = param_.residualAddScale;
    opParamFirst.preScale = firstLayerId + 1;
    opParamFirst.postScale = firstLayerId + 1;
    opParamFirst.transKey = param_.transKey;
    opParamFirst.model = "chatglm2_6b";  
    firstLayerNode.operation = std::make_shared<ChatGlm2LayerEncoderOperation>(opParamFirst);
    firstLayerNode.inTensors.resize(firstLayerNode.operation->GetInTensorCount());
    firstLayerNode.outTensors.resize(firstLayerNode.operation->GetOutTensorCount());
    size_t inTensorId = 0;
    firstLayerNode.inTensors.at(inTensorId++) = firstInTensor;
    for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_FIRST_LAYER; ++weightTensorId) {
        firstLayerNode.inTensors.at(inTensorId++) = 
            &graph_.weightTensors.at(firstLayerId * WEIGHT_COUNT_FIRST_LAYER + weightTensorId);
    }
    firstLayerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ROPECACHE);
    firstLayerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);

    firstLayerNode.outTensors = {&graph_.internalTensors.at(firstLayerId), &graph_.outTensors.at(firstLayerId + 1),
                                    &graph_.outTensors.at(firstLayerId + 1 + param_.layerNum)};
    firstInTensor = firstLayerNode.outTensors.at(0);
    int nodeId = 1;
    for (int layerId = 1; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        ChatGlm2QuantLayerParam opParam;
        opParam.numHeadsPerPartition = param_.numHeadsPerPartition;
        opParam.numGroupsPerPartition = param_.numGroupsPerPartition;
        opParam.hiddenSizePerHead = param_.hiddenSizePerHead;
        opParam.layerId = layerId;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.preScale = layerId + 1;
        opParam.postScale = layerId + 1;
        opParam.transKey = param_.transKey;
        opParam.model = "chatglm2_6b";
        opParam.qkvInputScale = param_.qkvInputScale[layerId];
        opParam.qkvInputOffset = param_.qkvInputOffset[layerId];
        opParam.denseInputScale = param_.denseInputScale[layerId];
        opParam.denseInputOffset = param_.denseInputOffset[layerId];
        opParam.selfLnInputScale = param_.selfLnInputScale[layerId];
        opParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
        opParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
        opParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];

        layerNode.operation = std::make_shared<ChatGlm2QuantLayerEncoderOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
        layerNode.outTensors.resize(layerNode.operation->GetOutTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = 
                &graph_.weightTensors.at(WEIGHT_COUNT_FIRST_LAYER + (layerId-1) * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ROPECACHE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BETA);

        layerNode.outTensors = {&graph_.internalTensors.at(layerId), &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    RmsNormParam finalNormParam = {param_.rmsNormEps};
    finalNormNode.operation = std::make_shared<RmsNormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.outTensors.at(0)};
}
} // namespace AclTransformer