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
#include "chatglm6b_encoder_quant_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "models/chatglm6b/chatglm6blayer_encoder_quant_operation.h"
#include "models/chatglm6b/chatglm6blayer_encoder_first_quant_operation.h"
#include "models/chatglm6b/chatglm6blayer_encoder_last_quant_operation.h"

namespace AclTransformer {
const size_t WEIGHT_COUNT_PER_LAYER = 16;

enum InTensorId {
    IN_TENSOR_HIDDEN_STATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_NUMS
};

void ChatGlm6BEncoderQuantTorch::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    qkvInputScale = paramJson["qkvInputScale"].get<std::vector<float>>();
    qkvInputOffset = paramJson["qkvInputOffset"].get<std::vector<int>>();
    denseInputScale = paramJson["denseInputScale"].get<std::vector<float>>();
    denseInputOffset = paramJson["denseInputOffset"].get<std::vector<int>>();
    selfLnInputScale = paramJson["selfLnInputScale"].get<std::vector<float>>();
    selfLnInputOffset = paramJson["selfLnInputOffset"].get<std::vector<int>>();
    ffnOutInputScale = paramJson["ffnOutInputScale"].get<std::vector<float>>();
    ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<std::vector<int>>();
    ASD_LOG(INFO) << "ChatGlm6BEncoderQuantTorch param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", transKey:" << transKey << ", dk:" << dk << ", layerNum:" << layerNum
                  << ", residualAddScale:" << residualAddScale;
}

ChatGlm6BEncoderQuantTorch::ChatGlm6BEncoderQuantTorch(const std::string &param) : Model("ChatGlm6BEncoderQuantTorch", param)
{
    param_.FromString(param);
}

ChatGlm6BEncoderQuantTorch::~ChatGlm6BEncoderQuantTorch() {}

uint64_t ChatGlm6BEncoderQuantTorch::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t ChatGlm6BEncoderQuantTorch::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status ChatGlm6BEncoderQuantTorch::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = inTensors.at(0).desc;

    for (size_t keyId = 0; keyId < param_.layerNum; ++keyId) {
        outTensorDescs.at(1 + keyId) = inTensors.at(0).desc;
        outTensorDescs.at(1 + keyId).dims.at(2) = param_.headNum;
        outTensorDescs.at(1 + keyId).dims.push_back(param_.dk);
    }
    for (size_t valueId = 0; valueId < param_.layerNum; ++valueId) {
        outTensorDescs.at(1 + param_.layerNum + valueId) = outTensorDescs.at(1 + valueId);
    }
    return AsdOps::Status::OkStatus(); 
}

void ChatGlm6BEncoderQuantTorch::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum;
    graph_.weightTensors.resize(weightTensorSize);

    // to be modified
    graph_.inTensors.resize(IN_TENSOR_NUMS);
    graph_.outTensors.resize(1 + 2 * param_.layerNum);

    const int nodeSize = param_.layerNum;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(2 * graph_.nodes.size() - 2);

    int nodeId = 0;

    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(0);
    AsdOps::Tensor *firstResInTensor = &graph_.inTensors.at(0);
    ASD_LOG(INFO) << "First InTensor Set.";

    for (int layerId = 0; layerId < param_.layerNum - 1; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        ChatGlm6BLayerQuantParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.transKey = param_.transKey;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.qkvInputScale = param_.qkvInputScale[layerId];
        opParam.qkvInputOffset = param_.qkvInputOffset[layerId];
        opParam.denseInputScale = param_.denseInputScale[layerId];
        opParam.denseInputOffset = param_.denseInputOffset[layerId];
        opParam.selfLnInputScale = param_.selfLnInputScale[layerId];
        opParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
        opParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
        opParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];

        if (layerId == 0) {
            layerNode.operation = std::make_shared<ChatGlm6BLayerEncoderFirstQuantOperation>(opParam);
        } else {
            layerNode.operation = std::make_shared<ChatGlm6BLayerEncoderQuantOperation>(opParam);
        }
        
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);      // cosTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);      // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = firstResInTensor; // inres

        layerNode.outTensors = {&graph_.internalTensors.at(2 * layerId), &graph_.outTensors.at(1 + layerId), &graph_.outTensors.at(1 + layerId + param_.layerNum),
                                &graph_.internalTensors.at(1 + 2 * layerId)};

        firstInTensor = layerNode.outTensors.at(0);
        firstResInTensor = layerNode.outTensors.at(3);
        ASD_LOG(INFO) << "Internal InTensor Set. OK ";
    }

    size_t lastInTensorId = 0;
    size_t lastLayerId = param_.layerNum - 1;
    auto &layerNode = graph_.nodes.at(nodeId++);

    ChatGlm6BLayerQuantParam lastOpParam;
    lastOpParam.layerNormEps = param_.layerNormEps;
    lastOpParam.headNum = param_.headNum;
    lastOpParam.transKey = param_.transKey;
    lastOpParam.dk = param_.dk;
    lastOpParam.layerId = lastLayerId;
    lastOpParam.residualAddScale = param_.residualAddScale;
    lastOpParam.qkvInputScale = param_.qkvInputScale[lastLayerId];
    lastOpParam.qkvInputOffset = param_.qkvInputOffset[lastLayerId];
    lastOpParam.denseInputScale = param_.denseInputScale[lastLayerId];
    lastOpParam.denseInputOffset = param_.denseInputOffset[lastLayerId];
    lastOpParam.selfLnInputScale = param_.selfLnInputScale[lastLayerId];
    lastOpParam.selfLnInputOffset = param_.selfLnInputOffset[lastLayerId];
    lastOpParam.ffnOutInputScale = param_.ffnOutInputScale[lastLayerId];
    lastOpParam.ffnOutInputOffset = param_.ffnOutInputOffset[lastLayerId];

    layerNode.operation = std::make_shared<ChatGlm6BLayerEncoderLastQuantOperation>(lastOpParam);

    layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
    layerNode.inTensors.at(lastInTensorId++) = firstInTensor;

    for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
        layerNode.inTensors.at(lastInTensorId++) = &graph_.weightTensors.at(lastLayerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
    }

    layerNode.inTensors.at(lastInTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
    layerNode.inTensors.at(lastInTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);      // cosTable
    layerNode.inTensors.at(lastInTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);      // sinTable
    layerNode.inTensors.at(lastInTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
    layerNode.inTensors.at(lastInTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
    layerNode.inTensors.at(lastInTensorId++) = firstResInTensor; // inres

    ASD_LOG(INFO) << "Build Graph finished.";

    layerNode.outTensors = {&graph_.outTensors.at(0), 
                            &graph_.outTensors.at(1 + lastLayerId), &graph_.outTensors.at(1 + lastLayerId + param_.layerNum)};
}

AsdOps::Status ChatGlm6BEncoderQuantTorch::ParseVarintPackParam(const std::string &param, int nodeId,
                                                                        AsdOps::Any &variantPackParam)
{
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
