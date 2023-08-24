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
#include "gptneox20b_decoder_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "models/gptneox20b/gptneox20blayer_embedding_operation.h"
#include "models/gptneox20b/gptneox20blayer_decoder_flashattention_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 12;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 1;

enum InTensorId {
    IN_TENSOR_INPUTIDS = 0,
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

void GptNeox20BDecoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rotaryPct = paramJson["rotaryPct"].get<float>();
    if (paramJson.contains("beginNormAxis")) {
        beginNormAxis = paramJson["beginNormAxis"].get<int>();
    }
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    ASD_LOG(INFO) << "GptNeox20BDecoderModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", transKey:" << transKey << ", dk:" << dk << ", layerNum:" << layerNum
                  << ", rotaryPct:" << rotaryPct << ", beginNormAxis:" << beginNormAxis
                  << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen;
}

GptNeox20BDecoderModel::GptNeox20BDecoderModel(const std::string &param): Model("GptNeox20BDecoderModel", param)
{
    param_.FromString(param);
}

GptNeox20BDecoderModel::~GptNeox20BDecoderModel(){}

uint64_t GptNeox20BDecoderModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t GptNeox20BDecoderModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status GptNeox20BDecoderModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1],
                                 graph_.weightTensors.at(0).desc.dims[1]};
    return AsdOps::Status::OkStatus();
}

void GptNeox20BDecoderModel::BuildGraph()
{
    const int weightTensorSize =
        WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(1);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    ASD_LOG(INFO) << "GptNeox20BDecoderModel nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() + 1);

    int nodeId = 0;
    auto &embeddingNode = graph_.nodes.at(nodeId++);
    embeddingNode.operation = std::make_shared<GptNeox20BLayerEmbeddingOperation>(GptNeox20BLayerEmbeddingParam());
    embeddingNode.inTensors.resize(embeddingNode.operation->GetInTensorCount());
    embeddingNode.outTensors.resize(embeddingNode.operation->GetOutTensorCount());
    embeddingNode.inTensors = {&graph_.weightTensors.at(0),
                               &graph_.inTensors.at(IN_TENSOR_INPUTIDS),
                               &graph_.inTensors.at(IN_TENSOR_COSTABLE),
                               &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                               &graph_.inTensors.at(IN_TENSOR_POSITIONID)
    };
    embeddingNode.outTensors = {&graph_.internalTensors.at(0),
                                &graph_.internalTensors.at(1),
                                &graph_.internalTensors.at(2)
    };

    AsdOps::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    AsdOps::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    AsdOps::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        GptNeox20BLayerDecoderFlashAttentionParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.transKey = param_.transKey;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.rotaryPct = param_.rotaryPct;
        opParam.tokenOffset = param_.tokenOffset;
        opParam.seqLen = param_.seqLen;
        layerNode.operation = std::make_shared<GptNeox20BLayerDecoderFlashAttentionOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;      // cosEmbed
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;      // sinEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTKEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTVALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    NormParam finalNormParam = {param_.layerNormEps, 2, 2};
    finalNormNode.operation = std::make_shared<NormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId = graph_.weightTensors.size() - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.outTensors.at(0)};
}

AsdOps::Status GptNeox20BDecoderModel::ParseVarintPackParam(const std::string &param, int nodeId,
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
    detailParam.layerId = 0;
    ASD_LOG(INFO) << "GptNeox20BDecoderModel SelfAttentionKvCacheFusionVariantPackParam tokenOffset:"
                  << detailParam.tokenOffset << ", seqLen:" << detailParam.seqLen
                  << ", layerId:" << detailParam.layerId;

    variantPackParam = detailParam;

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer