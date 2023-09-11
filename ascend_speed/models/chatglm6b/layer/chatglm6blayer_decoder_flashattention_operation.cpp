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
#include "chatglm6blayer_decoder_flashattention_operation.h"
#include "chatglm6b/operation/chatglm6b_position_embedding_rope_operation.h"

namespace atb_speed {

enum Chatglm6BLayerDecoderFlashAttentionTensorId {
    IN_HIDDENSTATES_ID = 0,
    IN_NORMWEIGHT_ID,
    IN_NORMBIAS_ID,
    IN_QKVMIXEDWEIGHT_ID,
    IN_QKVMIXEDBIAS_ID,
    IN_SELFOUTLINEARWEIGHT_ID,
    IN_SELFOUTLINEARBIAS_ID,
    IN_SELFOUTNORMWEIGHT_ID,
    IN_SELFOUTNORMBIAS_ID,
    IN_FFNLINEARWEIGHT_ID,
    IN_FFNLINEARBIAS_ID,
    IN_FFNOUTLINEARWEIGHT_ID,
    IN_FFNOUTLINEARBIAS_ID,
    IN_POSITIONIDS_ID,
    IN_COSTABLE_ID,
    IN_SINTABLE_ID,
    IN_ATTENTIONMASK_ID,
    IN_CACHEK_ID,
    IN_CACHEV_ID,
    IN_TOKENOFFSET_ID,
    IN_SEQLEN_ID,
    IN_LAYERID_ID,
    OUT_LAYEROUT_ID,
    INTERMEDIATE_INPUTNORMOUT_ID,
    INTERMEDIATE_MIXEDLINEAROUTQKV_ID,
    INTERMEDIATE_POSITIONEMBEDQ_ID,
    INTERMEDIATE_POSITIONEMBEDK_ID,
    INTERMEDIATE_VALUE_ID,
    INTERMEDIATE_SELFOUT_ID,
    INTERMEDIATE_SELFLINEAROUT_ID,
    INTERMEDIATE_SELFRESIDUALOUT_ID,
    INTERMEDIATE_SELFADDOUT_ID,
    INTERMEDIATE_SELFNORMOUT_ID,
    INTERMEDIATE_FFNOUT,
    INTERMEDIATE_FFNLINEAROUT_ID,
    INTERMEDIATE_FFNRESIDUALOUT_ID
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 13;
static const uint64_t NODE_COUNT = 12;

atb::Status CreateChatGlm6BLayerDecoderFlashAttentionOperation(const ChatGlm6BLayerDecoderFlashAttentionParam &param,
                                                               atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &inputNormNode = opGraph.nodes.at(nodeId++);
    auto &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);
    auto &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
    auto &selfAttentionFusionNode = opGraph.nodes.at(nodeId++);
    auto &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    auto &selfResidualNode = opGraph.nodes.at(nodeId++);
    auto &selfAddNode = opGraph.nodes.at(nodeId++);
    auto &selfNormNode = opGraph.nodes.at(nodeId++);
    auto &ffnNode = opGraph.nodes.at(nodeId++);
    auto &ffnLinearNode = opGraph.nodes.at(nodeId++);
    auto &ffnResidualNode = opGraph.nodes.at(nodeId++);
    auto &ffnAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::LayerNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    inputNormParam.normParam.epsilon = param.layerNormEps;
    inputNormParam.normParam.beginNormAxis = 2;
    inputNormParam.normParam.beginParamsAxis = 1;
    CreateOp(inputNormParam, &inputNormNode.op);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_NORMWEIGHT_ID, IN_NORMBIAS_ID};
    inputNormNode.outTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID};

    atb::infer::LinearParam mixdQkvLinearParam;
    CreateOp(mixdQkvLinearParam, &mixdQkvLinearNode.op);
    mixdQkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID, IN_QKVMIXEDWEIGHT_ID, IN_QKVMIXEDBIAS_ID};
    mixdQkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID};

    ChatGlm6BPositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = param.headNum;
    CreateChatGlm6BPositionEmbeddingOperation(positionEmbeddingParam, &positionEmbeddingNode.op);
    positionEmbeddingNode.inTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID, IN_POSITIONIDS_ID, IN_COSTABLE_ID,
                                         IN_SINTABLE_ID, IN_SEQLEN_ID};
    positionEmbeddingNode.outTensorIds = {INTERMEDIATE_POSITIONEMBEDQ_ID, INTERMEDIATE_POSITIONEMBEDK_ID,
                                          INTERMEDIATE_VALUE_ID};

    atb::infer::SelfAttentionFusionParam selfAttentionFusionParam;
    selfAttentionFusionParam.dk = param.dk;
    selfAttentionFusionParam.headNum = param.headNum;
    selfAttentionFusionParam.layerId = param.layerId;
    selfAttentionFusionParam.qScaleFlag = true;
    CreateOp(selfAttentionFusionParam, &selfAttentionFusionNode.op);
    selfAttentionFusionNode.inTensorIds = {INTERMEDIATE_POSITIONEMBEDQ_ID,
                                           INTERMEDIATE_POSITIONEMBEDK_ID,
                                           INTERMEDIATE_VALUE_ID,
                                           IN_CACHEK_ID,
                                           IN_CACHEV_ID,
                                           IN_ATTENTIONMASK_ID,
                                           IN_TOKENOFFSET_ID,
                                           IN_SEQLEN_ID,
                                           IN_LAYERID_ID};
    selfAttentionFusionNode.outTensorIds = {INTERMEDIATE_SELFOUT_ID};

    atb::infer::LinearParam selfOutLinearParam;
    CreateOp(selfOutLinearParam, &selfOutLinearNode.op);
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFOUT_ID, IN_SELFOUTLINEARWEIGHT_ID, IN_SELFOUTLINEARBIAS_ID};
    selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT_ID};

    atb::infer::ElewiseParam selfResidualParam;
    selfResidualParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    selfResidualParam.mulsParam.varAttr = param.residualAddScale;
    CreateOp(selfResidualParam, &selfResidualNode.op);
    selfResidualNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID};
    selfResidualNode.outTensorIds = {INTERMEDIATE_SELFRESIDUALOUT_ID};

    atb::infer::ElewiseParam selfAddParam;
    selfAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOp(selfAddParam, &selfAddNode.op);
    selfAddNode.inTensorIds = {INTERMEDIATE_SELFRESIDUALOUT_ID, INTERMEDIATE_SELFLINEAROUT_ID};
    selfAddNode.outTensorIds = {INTERMEDIATE_SELFADDOUT_ID};

    atb::infer::LayerNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    selfNormParam.normParam.epsilon = param.layerNormEps;
    selfNormParam.normParam.beginNormAxis = 2;
    selfNormParam.normParam.beginParamsAxis = 1;
    CreateOp(selfNormParam, &selfNormNode.op);
    selfNormNode.inTensorIds = {INTERMEDIATE_SELFADDOUT_ID, IN_SELFOUTNORMWEIGHT_ID, IN_SELFOUTNORMBIAS_ID};
    selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT_ID};

    atb::infer::LinearActivationParam ffnParam;
    CreateOp(ffnParam, &ffnNode.op);
    ffnNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID, IN_FFNLINEARWEIGHT_ID, IN_FFNLINEARBIAS_ID};
    ffnNode.outTensorIds = {INTERMEDIATE_FFNOUT};

    atb::infer::LinearParam ffnLinearParam;
    CreateOp(ffnLinearParam, &ffnLinearNode.op);
    ffnLinearNode.inTensorIds = {INTERMEDIATE_FFNOUT, IN_FFNOUTLINEARWEIGHT_ID, IN_FFNOUTLINEARBIAS_ID};
    ffnLinearNode.outTensorIds = {INTERMEDIATE_FFNLINEAROUT_ID};

    atb::infer::ElewiseParam ffnResidualParam;
    ffnResidualParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    ffnResidualParam.mulsParam.varAttr = param.residualAddScale;
    CreateOp(ffnResidualParam, &ffnResidualNode.op);
    ffnResidualNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID};
    ffnResidualNode.outTensorIds = {INTERMEDIATE_FFNRESIDUALOUT_ID};

    atb::infer::ElewiseParam ffnAddParam;
    ffnAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOp(ffnAddParam, &ffnAddNode.op);
    ffnAddNode.inTensorIds = {INTERMEDIATE_FFNRESIDUALOUT_ID, INTERMEDIATE_FFNLINEAROUT_ID};
    ffnAddNode.outTensorIds = {OUT_LAYEROUT_ID};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOp(opGraph, operation);
    return atb::NO_ERROR;
}

ChatGlm6BLayerDecoderFlashAttentionBinder::ChatGlm6BLayerDecoderFlashAttentionBinder() {}

ChatGlm6BLayerDecoderFlashAttentionBinder::~ChatGlm6BLayerDecoderFlashAttentionBinder() {}

void ChatGlm6BLayerDecoderFlashAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int32_t>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }

    layerId_ = paramJson["layerId"].get<int32_t>();
}

void ChatGlm6BLayerDecoderFlashAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET_ID).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN_ID).hostData = seqLen_.data();
    variantPack.inTensors.at(IN_LAYERID_ID).hostData = &layerId_;
}
} // namespace atb_speed