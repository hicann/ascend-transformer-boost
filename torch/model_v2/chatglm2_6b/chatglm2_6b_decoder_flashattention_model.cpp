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
#include "chatglm2_6b_decoder_flashattention_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "models/chatglm2_6b/chatglm2_6blayer_decoder_flashattention_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 7;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_AFTER_LAYER = 1;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_ROPEEMB,
    IN_TENSOR_PRESENTKEY,
    IN_TENSOR_PRESENTVALUE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_MAX,
};

void ChatGlm2DecoderFlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    headNum = paramJson["headNum"].get<int>();
    numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    transKey = paramJson["transKey"].get<bool>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int32_t>());
    }
    
    ASD_LOG(INFO) << "ChatGlm2DecoderFlashAttentionModel param rmsNormEps: " << rmsNormEps
                  << "numHeadsPerPartition: " << numHeadsPerPartition
                  << "hiddenSizePerHead: "<< hiddenSizePerHead
                  << "numGroupsPerPartition: "<< numGroupsPerPartition
                  << "transKey: " << transKey
                  << "residualAddScale: " << residualAddScale
                  << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen;;
}

ChatGlm2DecoderFlashAttentionModel::ChatGlm2DecoderFlashAttentionModel(const std::string &param) : Model("ChatGlm2DecoderModel", param)
{
    param_.FromString(param);
}

ChatGlm2DecoderFlashAttentionModel::~ChatGlm2DecoderFlashAttentionModel() {}

uint64_t ChatGlm2DecoderFlashAttentionModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t ChatGlm2DecoderFlashAttentionModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status ChatGlm2DecoderFlashAttentionModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                                std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

void ChatGlm2DecoderFlashAttentionModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(1);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(0);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        ChatGlm2LayerDecoderFlashAttentionParam opParam;

        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.headNum = param_.headNum;
        opParam.is2d = true;
        opParam.numHeadsPerPartition = param_.numHeadsPerPartition;
        opParam.hiddenSizePerHead = param_.hiddenSizePerHead;
        opParam.numGroupsPerPartition = param_.numGroupsPerPartition;
        opParam.transKey = param_.transKey;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.model = "chatglm2_6b";
        opParam.seqLen = param_.seqLen;
        opParam.tokenOffset = param_.tokenOffset;

        layerNode.operation = std::make_shared<ChatGlm2LayerDecoderFlashAttentionOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ROPEEMB);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PRESENTKEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PRESENTVALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        layerNode.outTensors = {&graph_.internalTensors.at(layerId)};
        firstInTensor = layerNode.outTensors.at(0); 
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    RmsNormParam finalNormParam = {param_.rmsNormEps};
    finalNormNode.operation = std::make_shared<RmsNormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.outTensors.at(0)};
}

AsdOps::Status ChatGlm2DecoderFlashAttentionModel::ParseVarintPackParam(const std::string &param, int nodeId, AsdOps::Any &variantPackParam)
{
    AclTransformer::SelfAttentionKvCacheFusionVariantPackParam opParam;
    nlohmann::json paramJson = nlohmann::json::parse(param);
    for (auto item : paramJson["tokenOffset"]) {
        opParam.tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        opParam.seqLen.push_back(item.get<int>());
    }
    opParam.layerId = nodeId;
    ASD_LOG(INFO) << "ChatGlm2DecoderFlashAttentionModel SelfAttentionKvCacheFusionVariantPackParam tokenOffset: "
                  << opParam.tokenOffset << ", seqLen" << opParam.seqLen
                  << ", layerId" << opParam.layerId;
    variantPackParam = opParam;

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
