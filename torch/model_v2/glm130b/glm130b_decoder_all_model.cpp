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
#include "glm130b_decoder_all_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/lm_head_parallel_operation.h"
#include "acltransformer/ops/word_embedding_parallel_operation.h"
#include "models/glm130b/glm130blayer_decoder_with_fusion_operation.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 12;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 0;
const int FINALNORMNODE_WEIGHT_COUNT = 3;   // change to 3 includes final forward weight
const int OPERATION_COUNT_BEFORE_LAYER = 2;
const int OPERATION_COUNT_AFTER_LAYER = 2;   // final norm and lm head

enum InTensorId {
    IN_TENSOR_HIDDENSTATSE = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PASTKEY,
    IN_TENSOR_PASTVALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_LAYERID_BASE,
};

enum InternelTensorId {
    INTERNEL_TENSOR_COS = 0,
    INTERNEL_TENSOR_SIN,
    INTERNEL_TENSOR_LAYEROUT_BASE,
};

void Glm130BDecoderAllModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    dk = paramJson["dk"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    backend = paramJson["backend"].get<std::string>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    ASD_LOG(INFO) << "Glm130BDecoderAllModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", transKey:" << transKey << ", dk:" << dk << ", layerNum:" << layerNum
                  << ", residualAddScale:" << residualAddScale << ", rank:" << rank << ", rankSize:" << rankSize
                  << ", backend:" << backend << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen;
}

Glm130BDecoderAllModel::Glm130BDecoderAllModel(const std::string &param)
    : Model("Glm130BDecoderAllModel", param)
{
    param_.FromString(param);
}

Glm130BDecoderAllModel::~Glm130BDecoderAllModel() {}

uint64_t Glm130BDecoderAllModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t Glm130BDecoderAllModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status Glm130BDecoderAllModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                     std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[1], inTensors.at(0).desc.dims[0],
                                graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dims[0] * param_.rankSize};

    return AsdOps::Status::OkStatus();
}

void Glm130BDecoderAllModel::BuildGraph()
{
    const int weightTensorSize =
        WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_LAYERID_BASE + param_.layerNum);
    graph_.outTensors.resize(1);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = INTERNEL_TENSOR_LAYEROUT_BASE + param_.layerNum;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &cosEmbeddingNode = graph_.nodes.at(nodeId++);
    cosEmbeddingNode.operation = std::make_shared<EmbeddingOperation>(EmbeddingParam());
    cosEmbeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_COSTABLE),
                                  &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    cosEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_COS)};

    auto &sinEmbeddingNode = graph_.nodes.at(nodeId++);
    sinEmbeddingNode.operation = std::make_shared<EmbeddingOperation>(EmbeddingParam());
    sinEmbeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_SINTABLE),
                                  &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    sinEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_SIN)};

    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(IN_TENSOR_HIDDENSTATSE);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        Glm130BLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.transKey = param_.transKey;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.tokenOffset = param_.tokenOffset;
        opParam.seqLen = param_.seqLen;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        layerNode.operation = std::make_shared<ChatGlm130BLayerDecoderFusionOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_COS);     // cosTable
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNEL_TENSOR_SIN);     // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTKEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTVALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_LAYERID_BASE + layerId);
        layerNode.inTensorViewFuncs.resize(IN_TENSOR_LAYERID_BASE + 1);
        layerNode.inTensorViewFuncs.at(2) = [] (const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
            newDims = {oldDims.at(0), oldDims.at(1), 1, oldDims.at(2)};
        };
        layerNode.inTensorViewFuncs.at(3) = layerNode.inTensorViewFuncs.at(2);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_LAYEROUT_BASE + layerId)};
    

        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    NormParam finalNormParam = {param_.layerNormEps};
    finalNormNode.operation = std::make_shared<NormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId = finalLayerNormWeightTensorId + 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(internalTensorSize - 1)};

    auto &LmHeadParallelNode = graph_.nodes.at(nodeId++);
    LmHeadParallelParam LmHeadParam = {param_.rank, param_.rankSize, param_.rankRoot, param_.backend, param_.perm};
    LmHeadParallelNode.operation = std::make_shared<LmHeadParallelOperation>(LmHeadParam);
    const int finalForwardWeightTensorId = graph_.weightTensors.size() - 1;
    LmHeadParallelNode.inTensors = {&graph_.internalTensors.at(internalTensorSize - 1), &graph_.weightTensors.at(finalForwardWeightTensorId)};
    LmHeadParallelNode.outTensors = {&graph_.outTensors.at(0)};
}

AsdOps::Status Glm130BDecoderAllModel::ParseVarintPackParam(const std::string &param, int nodeId,
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
    detailParam.layerId = nodeId - OPERATION_COUNT_BEFORE_LAYER;
    ASD_LOG(INFO) << "Glm130BDecoderAllModel SelfAttentionKvCacheFusionVariantPackParam tokenOffset:"
                  << detailParam.tokenOffset << ", seqLen:" << detailParam.seqLen
                  << ", layerId:" << detailParam.layerId;

    variantPackParam = detailParam;

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer