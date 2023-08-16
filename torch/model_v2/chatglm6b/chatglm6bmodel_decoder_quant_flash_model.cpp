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
#include "chatglm6bmodel_decoder_quant_flash_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "models/chatglm6b/chatglm6blayer_decoder_quant_flash_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_first_quant_flash_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_last_quant_flash_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 16;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PASTKEY,
    IN_TENSOR_PASTVALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX,
};

void ChatGlm6BDecoderQuantFlashModel::Param::FromString(const std::string &param)
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
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }

    
    ASD_LOG(INFO) << "ChatGlm6BDecoderQuantFlashModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", transKey:" << transKey << ", dk:" << dk << ", layerNum:" << layerNum
                  << ", residualAddScale:" << residualAddScale << ", qkvInputScale:" << qkvInputScale
                  << ", qkvInputOffset:" << qkvInputOffset << ", denseInputScale:" << denseInputScale
                  << ", selfLnInputScale:" << selfLnInputScale << ", selfLnInputOffset:" << selfLnInputOffset
                  << ", ffnOutInputScale:" << ffnOutInputScale << ", ffnOutInputOffset:" << ffnOutInputOffset
                  << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen;
}

ChatGlm6BDecoderQuantFlashModel::ChatGlm6BDecoderQuantFlashModel(const std::string &param) : Model("ChatGlm6BDecoderQuantFlashModel", param)
{
    param_.FromString(param);
}

ChatGlm6BDecoderQuantFlashModel::~ChatGlm6BDecoderQuantFlashModel() {}

uint64_t ChatGlm6BDecoderQuantFlashModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t ChatGlm6BDecoderQuantFlashModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status ChatGlm6BDecoderQuantFlashModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

void ChatGlm6BDecoderQuantFlashModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum; 
    graph_.weightTensors.resize(weightTensorSize);
    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(1);
    const int nodeSize = param_.layerNum;
    graph_.nodes.resize(nodeSize);
    graph_.internalTensors.resize(2 * graph_.nodes.size() - 1);

    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(IN_TENSOR_HIDDENSTATES);
    AsdOps::Tensor *firstResInTensor = &graph_.inTensors.at(IN_TENSOR_HIDDENSTATES);
    int nodeId = 0;
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        ChatGlm6BLayerQuantFlashParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.transKey = param_.transKey;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.tokenOffset = param_.tokenOffset;
        opParam.seqLen = param_.seqLen;
        opParam.qkvInputScale = param_.qkvInputScale[layerId];
        opParam.qkvInputOffset = param_.qkvInputOffset[layerId];
        opParam.denseInputScale = param_.denseInputScale[layerId];
        opParam.denseInputOffset = param_.denseInputOffset[layerId];
        opParam.selfLnInputScale = param_.selfLnInputScale[layerId];
        opParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
        opParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
        opParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];

        if (layerId == 0) {
            layerNode.operation = std::make_shared<ChatGlm6BLayerDecoderFirstQuantFlashOperation>(opParam);
        } else if (layerId == param_.layerNum - 1){
            layerNode.operation = std::make_shared<ChatGlm6BLayerDecoderLastQuantFlashOperation>(opParam);
        } else {
            layerNode.operation = std::make_shared<ChatGlm6BLayerDecoderQuantFlashOperation>(opParam);
        }
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);      // cosTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);      // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTKEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTVALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        layerNode.inTensors.at(inTensorId++) = firstResInTensor;
       
        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(2 * layerId), 
            &graph_.internalTensors.at(2 * layerId + 1)};
            firstInTensor = layerNode.outTensors.at(0);
            firstResInTensor = layerNode.outTensors.at(1);
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0)};
        }
    }
}

AsdOps::Status ChatGlm6BDecoderQuantFlashModel::ParseVarintPackParam(const std::string &param, int nodeId,
                                                           AsdOps::Any &variantPackParam)
{
    AclTransformer::SelfAttentionKvCacheFusionVariantPackParam detailParam;
    nlohmann::json paramJson = nlohmann::json::parse(param);
    for (auto item : paramJson["tokenOffset"]) {
        detailParam.tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        detailParam.seqLen.push_back(item.get<int>());
    }
    detailParam.layerId = nodeId;
    ASD_LOG(INFO) << "ChatGlm6BDecoderQuantFlashModel SelfAttentionKvCacheFusionVariantPackParam tokenOffset:"
                  << detailParam.tokenOffset << ", seqLen:" << detailParam.seqLen
                  << ", layerId:" << detailParam.layerId;

    variantPackParam = detailParam;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer