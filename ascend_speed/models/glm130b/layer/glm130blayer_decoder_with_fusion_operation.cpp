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
#include "glm130blayer_decoder_with_fusion_operation.h"

namespace atb_speed {
static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

atb::Status CreateGlm130BLayerDecoderFusionOperation(const Glm130BLayerParam &param, atb::Operation **operation)
{
#if 0
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &inputNormNode = opGraph.nodes.at(nodeId++);
    auto &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);
    auto &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
    auto &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    auto &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    auto &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    auto &selfNormNode = opGraph.nodes.at(nodeId++);
    auto &mlpNode = opGraph.nodes.at(nodeId++);
    auto &mlpLinearParallelNode = opGraph.nodes.at(nodeId++);
    auto &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer_old::NormParam inputNormParam;
    inputNormParam.layerNormEps = param.layerNormEps;
    CreateOp(inputNormParam, &inputNormNode.op);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_NORMWEIGHT_ID, IN_NORMBIAS_ID};
    inputNormNode.outTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID};

    atb::infer::LinearParam mixdQkvLinearParam;
    CreateOp(mixdQkvLinearParam, &mixdQkvLinearNode.op);
    mixdQkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID, IN_QKVMIXEDWEIGHT_ID, IN_QKVMIXEDBIAS_ID};
    mixdQkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID};

    atb::infer_old::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.is2d = false;
    positionEmbeddingParam.headNum = param.headNum / param.rankSize;
    positionEmbeddingParam.isFusion = true;
    positionEmbeddingParam.rotaryCoeff = 2;
    CreateOp(positionEmbeddingParam, &positionEmbeddingNode.op);
    positionEmbeddingNode.inTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID, IN_COS_ID, IN_SIN_ID,
                                         IN_SEQLEN_ID};
    positionEmbeddingNode.outTensorIds = {INTERMEDIATE_POSITIONEMBEDQ_ID, INTERMEDIATE_POSITIONEMBEDK_ID,
                                          INTERMEDIATE_VALUE_ID};

    atb::infer_old::SelfAttentionKvCacheFusionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param.headNum / param.rankSize;
    selfAttentionKvCacheParam.layerId = param.layerId;
    selfAttentionKvCacheParam.dk = param.dk;
    selfAttentionKvCacheParam.tokenOffset = param.tokenOffset;
    selfAttentionKvCacheParam.seqLen = param.seqLen;
    CreateOp(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.op);
    selfAttentionKvCacheNode.inTensorIds = {INTERMEDIATE_POSITIONEMBEDK_ID,
                                            INTERMEDIATE_VALUE_ID,
                                            IN_CACHEK_ID,
                                            IN_CACHEV_ID,
                                            INTERMEDIATE_POSITIONEMBEDQ_ID,
                                            IN_ATTENTIONMASK_ID,
                                            IN_TOKENOFFSET_ID,
                                            IN_SEQLEN_ID,
                                            IN_LAYERID_ID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMEDIATE_SELFOUT_ID};

    atb::infer_old::LinearParallelParam selfOutLinearParam;
    selfOutLinearParam.transWeight = false;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.rankRoot = 0;
    selfOutLinearParam.parallelType = "RowParallel";
    selfOutLinearParam.backend = param.backend;
    CreateOp(selfOutLinearParam, &selfOutLinearNode.op);
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFOUT_ID, IN_SELFOUTLINEARWEIGHT_ID, IN_SELFOUTLINEARBIAS_ID};
    selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT_ID};

    atb::infer_old::AddParam selfResidualAddParam;
    selfResidualAddParam.scale = param.residualAddScale;
    CreateOp(selfResidualAddParam, &selfResidualAddNode.op);
    selfResidualAddNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID, INTERMEDIATE_SELFLINEAROUT_ID};
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_SELFRESIDUALADDOUT_ID};

    atb::infer_old::NormParam selfNormParam;
    selfNormParam.layerNormEps = param.layerNormEps;
    CreateOp(selfNormParam, &selfNormNode.op);
    selfNormNode.inTensorIds = {INTERMEDIATE_SELFRESIDUALADDOUT_ID, IN_SELFOUTNORMWEIGHT_ID, IN_SELFOUTNORMBIAS_ID};
    selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT_ID};

    atb::infer_old::MlpParam mlpParam;
    mlpParam.model = "glm130b";
    CreateOp(mlpParam, &mlpNode.op);
    mlpNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID, IN_MLPLINEARWEIGHT_ID, IN_MLPLINEARBIAS_ID};
    mlpNode.outTensorIds = {INTERMEDIATE_MLPOUT};

    atb::infer_old::LinearParallelParam mlpLinearParam;
    mlpLinearParam.transWeight = false;
    mlpLinearParam.rank = param.rank;
    mlpLinearParam.rankSize = param.rankSize;
    mlpLinearParam.rankRoot = 0;
    mlpLinearParam.parallelType = "RowParallel";
    mlpLinearParam.backend = param.backend;
    CreateOp(mlpLinearParam, &mlpLinearParallelNode.op);
    mlpLinearParallelNode.inTensorIds = {INTERMEDIATE_MLPOUT, IN_MLPOUTLINEARWEIGHT_ID, IN_MLPOUTLINEARBIAS_ID};
    mlpLinearParallelNode.outTensorIds = {INTERMEDIATE_MLPLINEAROUT_ID};

    atb::infer_old::AddParam mlpResidualAddParam;
    mlpResidualAddParam.scale = param.residualAddScale;
    CreateOp(mlpResidualAddParam, &mlpResidualAddNode.op);
    mlpResidualAddNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID, INTERMEDIATE_MLPLINEAROUT_ID};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT_ID};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOp(opGraph, operation);
#endif
    return atb::NO_ERROR;
}

ChatGlm130BLayerDecoderFlashAttentionBinder::ChatGlm130BLayerDecoderFlashAttentionBinder() {}

ChatGlm130BLayerDecoderFlashAttentionBinder::~ChatGlm130BLayerDecoderFlashAttentionBinder() {}

void ChatGlm130BLayerDecoderFlashAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int32_t>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int64_t>());
    }

    layerId_ = paramJson["layerId"].get<int32_t>();
}

void ChatGlm130BLayerDecoderFlashAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t seqLenTensorId = 19;
    const uint32_t tokenOffsetTensorId = 20;
    const uint32_t layerIdTensorId = 21;
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;
}
} // namespace AclTransformer