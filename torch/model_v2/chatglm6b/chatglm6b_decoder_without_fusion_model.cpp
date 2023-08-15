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
#include "chatglm6b_decoder_without_fusion_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "models/chatglm6b/chatglm6blayer_decoder_without_fusion_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 12;

enum InTensorId {
    IN_TENSOR_HIDDEN_STATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PASTK_V_START
};

void ChatGlm6BDecoderWithoutFusionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    if (paramJson.contains("beginNormAxis")) {
        beginNormAxis = paramJson["beginNormAxis"].get<int>();
    }
    if (paramJson.contains("beginParamsAxis")) {
        beginParamsAxis = paramJson["beginParamsAxis"].get<int>();
    }
    ASD_LOG(INFO) << "ChatGlm6BDecoderWithoutFusionModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", transKey:" << transKey << ", dk:" << dk << ", layerNum:" << layerNum
                  << ", residualAddScale:" << residualAddScale << ", beginNormAxis:" << beginNormAxis
                  << ", beginParamsAxis:" << beginParamsAxis;
}

ChatGlm6BDecoderWithoutFusionModel::ChatGlm6BDecoderWithoutFusionModel(const std::string &param) : Model("ChatGlm6BDecoderWithoutFusionModel", param)
{
    param_.FromString(param);
}

ChatGlm6BDecoderWithoutFusionModel::~ChatGlm6BDecoderWithoutFusionModel() {}

uint64_t ChatGlm6BDecoderWithoutFusionModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t ChatGlm6BDecoderWithoutFusionModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status ChatGlm6BDecoderWithoutFusionModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                                std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    const AsdOps::Tensor &keyTensor = inTensors.at(IN_TENSOR_PASTK_V_START);
    const AsdOps::Tensor &valueTensor = inTensors.at(IN_TENSOR_PASTK_V_START + param_.layerNum);

    outTensorDescs.at(0) = inTensors.at(0).desc;

    for (size_t keyId = 0; keyId < param_.layerNum; ++keyId) {
        outTensorDescs.at(1 + 2 * keyId) = keyTensor.desc;
        outTensorDescs.at(1 + 2 * keyId).dims.at(0) += 1;   
    }
    for (size_t valueId = 0; valueId < param_.layerNum; ++valueId) {
        outTensorDescs.at(2 + 2 * valueId) = valueTensor.desc;
        outTensorDescs.at(2 + 2 * valueId).dims.at(0) += 1;   
    }
    return AsdOps::Status::OkStatus();
}

void ChatGlm6BDecoderWithoutFusionModel::BuildGraph()
{
    ASD_LOG(INFO) << "Build Graph Start.";
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum; 
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_PASTK_V_START + 3 * param_.layerNum);
    graph_.outTensors.resize(1 + 2 * param_.layerNum);

    const int nodeSize = param_.layerNum;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;   

    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(0);
    ASD_LOG(INFO) << "First InTensor Set.";

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        ChatGlm6BLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.transKey = param_.transKey;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.beginNormAxis = param_.beginNormAxis;
        opParam.beginParamsAxis = param_.beginParamsAxis;
        layerNode.operation = std::make_shared<ChatGlm6BLayerDecoderWithoutFusionOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;   // hidden states tensor

        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }

        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);       // positionIdTensor
        
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);       // costable
        
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);       // sintable
        
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);  // attentionMaskTensor
        
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTK_V_START + layerId);
        
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTK_V_START + param_.layerNum + layerId);
        
        
        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(layerId),
                                    &graph_.outTensors.at(2 * layerId + 1),
                                    &graph_.outTensors.at(2 * layerId + 2)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0),
                                    &graph_.outTensors.at(2 * layerId + 1),
                                    &graph_.outTensors.at(2 * layerId + 2)};
        }

        firstInTensor = layerNode.outTensors.at(0);
    }
    ASD_LOG(INFO) << "Build Graph finished.";
}

AsdOps::Status ChatGlm6BDecoderWithoutFusionModel::ParseVarintPackParam(const std::string &param, int nodeId,
                                                                        AsdOps::Any &variantPackParam)
{
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer