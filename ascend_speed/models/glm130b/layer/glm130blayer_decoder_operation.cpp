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
#include "glm130blayer_decoder_operation.h"

namespace atb_speed {
enum glm130BLayerDecoderTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTNORMBIAS,
    IN_MLPLINEARWEIGHT,
    IN_MLPLINEARBIAS,
    IN_MLPOUTLINEARWEIGHT,
    IN_MLPOUTLINEARBIAS,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    OUT_GLMLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_VALUE,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
    INTERMIDATE_MLPLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

atb::Status CreateGlm130BLayerDecoderFlashAttentionOperation(const Glm130BLayerParam &param, atb::Operation **operation)
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
    auto &selfOutLinearParallelNode = opGraph.nodes.at(nodeId++);
    auto &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    auto &selfNormNode = opGraph.nodes.at(nodeId++);
    auto &mlpNode = opGraph.nodes.at(nodeId++);
    auto &mlpLinearParallelNode = opGraph.nodes.at(nodeId++);
    auto &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer_old::NormParam inputNormParam;
    inputNormParam.layerNormEps = param.layerNormEps;
    CreateOp(inputNormParam, &inputNormNode.op);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::LinearParam mixdQkvLinearParam;
    CreateOp(mixdQkvLinearParam, &mixdQkvLinearNode.op);
    mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS};
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    atb::infer_old::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.is2d = false;
    positionEmbeddingParam.headNum = param.headNum / param.rankSize;
    CreateOp(positionEmbeddingParam, &positionEmbeddingNode.op);
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE};

    atb::infer_old::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.transKey = param.transKey;
    selfAttentionKvCacheParam.dk = param.dk;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.layerId = param.layerId;
    CreateOp(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.op);
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_VALUE,
                                            IN_ATTENTIONMASK,
                                            IN_PASTKEY,
                                            IN_PASTVALUE};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    atb::infer_old::LinearParallelParam selfOutLinearParam;
    selfOutLinearParam.transWeight = false;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.rankRoot = 0;
    selfOutLinearParam.parallelType = "RowParallel";
    selfOutLinearParam.backend = param.backend;
    CreateOp(selfOutLinearParam, &selfOutLinearParallelNode.op);
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    atb::infer_old::AddParam selfResidualAddParam;
    selfResidualAddParam.scale = param.residualAddScale;
    CreateOp(selfResidualAddParam, &selfResidualAddNode.op);
    selfResidualAddNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    atb::infer_old::NormParam selfNormParam;
    selfNormParam.layerNormEps = param.layerNormEps;
    CreateOp(selfNormParam, &selfNormNode.op);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb::infer_old::MlpParam mlpParam;
    mlpParam.model = "glm130b";
    CreateOp(mlpParam, &mlpNode.op);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPLINEARWEIGHT, IN_MLPLINEARBIAS};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    atb::infer_old::LinearParallelParam mlpLinearParam;
    mlpLinearParam.transWeight = false;
    mlpLinearParam.rank = param.rank;
    mlpLinearParam.rankSize = param.rankSize;
    mlpLinearParam.rankRoot = 0;
    mlpLinearParam.parallelType = "RowParallel";
    mlpLinearParam.backend = param.backend;
    CreateOp(mlpLinearParam, &mlpLinearParallelNode.op);
    mlpLinearParallelNode.inTensorIds = {INTERMIDATE_MLPOUT, IN_MLPOUTLINEARWEIGHT, IN_MLPOUTLINEARBIAS};
    mlpLinearParallelNode.outTensorIds = {INTERMIDATE_MLPLINEAROUT};

    atb::infer_old::AddParam mlpResidualAddParam;
    mlpResidualAddParam.scale = param.residualAddScale;
    CreateOp(mlpResidualAddParam, &mlpResidualAddNode.op);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, INTERMIDATE_MLPLINEAROUT};
    mlpResidualAddNode.outTensorIds = {OUT_GLMLAYEROUT};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        const atb::TensorDesc &hiddenStatesTensorDesc = inTensorDescs.at(0);
        const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(IN_PASTKEY);
        const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(IN_PASTVALUE);
        outTensorDescs.at(0) = hiddenStatesTensorDesc;
        outTensorDescs.at(1) = keyTensorDesc;
        outTensorDescs.at(1).shape.dims[0] += 1;
        const size_t tensorId2 = 2;
        outTensorDescs.at(tensorId2) = valueTensorDesc;
        outTensorDescs.at(tensorId2).shape.dims[0] += 1;
        return atb::NO_ERROR;
    };

    atb::CreateOp(opGraph, operation);
#endif
    return atb::NO_ERROR;
}
} // namespace atb_speed