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
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "models/gptneox20b/gptneox20blayer_embedding_operation.h"
#include "llama13b_fusion_model.h"
#include "models/llama13b/llama13blayer_fusion_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 7;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;

enum InTensorId {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX
};

void Llama13BFusionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    model = paramJson["model"].get<std::string>();
    rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    ASD_LOG(INFO) << "Llama13BFusionModel param rmsNormEps:" << rmsNormEps
                  << ", headNum:" << headNum  
                  << ", dk:" << dk
                  << ", layerNum:" << layerNum
                  << ", model:" << model
                  << ", rotaryCoeff:" << rotaryCoeff
                  << ", tokenOffst:" << tokenOffset
                  << ", seqLen:" << seqLen;
}

Llama13BFusionModel::Llama13BFusionModel(const std::string &param) : Model("Llama13BFusionModel", param)
{
    param_.FromString(param);
}

Llama13BFusionModel::~Llama13BFusionModel() {}

uint64_t Llama13BFusionModel::GetInTensorCount() const
{
    return graph_.inTensors.size(); 
}

uint64_t Llama13BFusionModel::GetOutTensorCount() const
{
    return graph_.outTensors.size();
}

AsdOps::Status Llama13BFusionModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    ASD_LOG(INFO) << "Enter Llama13BFusionModel InferShape";
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(1).desc;
    outTensorDescs.at(0).dims = {inTensors.at(1).desc.dims[0], inTensors.at(1).desc.dims[1], outDim};

    return AsdOps::Status::OkStatus();
}

void Llama13BFusionModel::BuildGraph()
{
    ASD_LOG(INFO) << "Enter Llama13BFusionModel BuildGraph";
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
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

        LLaMA13BLayerFusionParam modelParam;
        modelParam.rmsNormEps = param_.rmsNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.layerId = param_.layerId;
        modelParam.tokenOffset = param_.tokenOffset;
        modelParam.seqLen = param_.seqLen;
        modelParam.rotaryCoeff = param_.rotaryCoeff;
        modelParam.model = param_.model;

        layerNode.operation = std::make_shared<LLaMA13BLayerFusionOperation>(modelParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    RmsNormParam finalNormParam = {param_.rmsNormEps};
    finalNormNode.operation = std::make_shared<RmsNormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT
                                            - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    LinearParam linearParam;
    linearParam.hasBias = false;
    outLinearNode.operation = std::make_shared<LinearOperation>(linearParam);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId,
                                &graph_.weightTensors.at(finalLinearWeightTensorId))};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};
}

AsdOps::Status Llama13BFusionModel::ParseVarintPackParam(const std::string &param, int nodeId, 
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
    ASD_LOG(INFO) << "Llama13BFusionModel SelfAttentionKvCacheFusionVariantPackParam tokenOffset:"
                  << detailParam.tokenOffset << ", seqLen:" << detailParam.seqLen
                  << ", layerId:" << detailParam.layerId;

    variantPackParam = detailParam;

    return AsdOps::Status::OkStatus();
}
}
