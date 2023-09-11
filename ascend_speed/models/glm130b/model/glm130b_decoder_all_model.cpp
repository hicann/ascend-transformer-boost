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
#include "glm130b/layer/glm130blayer_decoder_with_fusion_operation.h"
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>


namespace atb_speed {
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
    ATB_LOG(INFO) << "Glm130BDecoderAllModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
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

uint32_t Glm130BDecoderAllModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t Glm130BDecoderAllModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status Glm130BDecoderAllModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                     std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[2] = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0] * param_.rankSize;

    return atb::NO_ERROR;
}

void Glm130BDecoderAllModel::BuildGraph()
{
#if 0
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
    atb::infer_old::EmbeddingParam ropeCosParam;
    atb::Operation *op = nullptr;
    atb::CreateOp(ropeCosParam, &op);
    cosEmbeddingNode.operation.reset(op);
    cosEmbeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_COSTABLE),
                                  &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    cosEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_COS)};

    auto &sinEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer_old::EmbeddingParam ropeSinParam;
    atb::CreateOp(ropeSinParam, &op);
    sinEmbeddingNode.operation.reset(op);
    sinEmbeddingNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_SINTABLE),
                                  &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    sinEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_SIN)};

    atb::Tensor *firstInTensor = &graph_.inTensors.at(IN_TENSOR_HIDDENSTATSE);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::Glm130BLayerParam opParam;
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
        CreateGlm130BLayerDecoderFusionOperation(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

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
        layerNode.inTensorReshapeFuncs.resize(IN_TENSOR_LAYERID_BASE + 1);
        layerNode.inTensorReshapeFuncs.at(2) = [] (const atb::Dims &oldDims, atb::Dims &newDims) {
            newDims.dims[0] = oldDims.dims[0];
            newDims.dims[1] = oldDims.dims[1];
            newDims.dims[2] = 1;
            newDims.dims[3] = oldDims.dims[2];
        };
        layerNode.inTensorReshapeFuncs.at(3) = layerNode.inTensorReshapeFuncs.at(2);
        layerNode.outTensors = {&graph_.internalTensors.at(INTERNEL_TENSOR_LAYEROUT_BASE + layerId)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer_old::NormParam finalNormParam;
    finalNormParam.layerNormEps = param_.layerNormEps;
    atb::CreateOp(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId = finalLayerNormWeightTensorId + 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(internalTensorSize - 1)};

    auto &LmHeadParallelNode = graph_.nodes.at(nodeId++);
    atb::infer_old::LmHeadParallelParam lmHeadParam;
    lmHeadParam.rank = param_.rank;
    lmHeadParam.rankSize = param_.rankSize;
    lmHeadParam.rankRoot = param_.rankRoot;
    lmHeadParam.backend = param_.backend;
    lmHeadParam.perm = param_.perm;
    CreateOp(lmHeadParam, &op);
    LmHeadParallelNode.operation.reset(op);
    const int finalForwardWeightTensorId = graph_.weightTensors.size() - 1;
    LmHeadParallelNode.inTensors = {&graph_.internalTensors.at(internalTensorSize - 1), &graph_.weightTensors.at(finalForwardWeightTensorId)};
    LmHeadParallelNode.outTensors = {&graph_.outTensors.at(0)};
#endif
}

atb::Status Glm130BDecoderAllModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    
    ATB_LOG(INFO) << "ChatGlm130BDecoderModel ParseParam tokenOffset:" << tokenOffset_ << ", seqLen:" << seqLen_;

    return atb::NO_ERROR;
}

atb::Status Glm130BDecoderAllModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }

    layerId_ = nodeId - OPERATION_COUNT_BEFORE_LAYER;
    auto &node = graph_.nodes.at(nodeId);

    const uint32_t seqLenTensorId = Chatglm130BLayerDecoderFlashAttentionTensorId::IN_SEQLEN_ID;
    const uint32_t tokenOffsetTensorId = Chatglm130BLayerDecoderFlashAttentionTensorId::IN_TOKENOFFSET_ID;
    const uint32_t layerIdTensorId = Chatglm130BLayerDecoderFlashAttentionTensorId::IN_LAYERID_ID;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;

    return atb::NO_ERROR;
}
} // namespace atb_speed