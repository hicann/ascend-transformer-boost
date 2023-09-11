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
#include "llamalayer_fusion_parallel_operation.h"

namespace atb_speed {
enum LlamaLayerFusionParallelTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QMIXDWEIGHT,
    IN_KMIXDWEIGHT,
    IN_VMIXDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    OUT_LLAMA13BLAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
    INTERMIDATE_MLPLINEARPARALLELOUT,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 12;

atb::Status LlamaLayerFusionParallelOperation(const LlamaLayerFusionParallelParam &param,
                                                        atb::Operation **operation)
{
#if 0
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQLinearNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mixdKLinearNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mixdVLinearNode  = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpLinearParallelNode   = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode   = opGraph.nodes.at(nodeId++);

    atb::infer_old::RmsNormParam inputNormParam;
    inputNormParam.rmsNormEps = param.rmsNormEps;
    CreateOp(inputNormParam, &inputNormNode.op);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    atb::infer::LinearParam mixdQLinearParam;
    mixdQLinearParam.transposeA = false;
    mixdQLinearParam.transposeB = false;
    mixdQLinearParam.hasBias = false;
    CreateOp(mixdQLinearParam, &mixdQLinearNode.op);
    mixdQLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QMIXDWEIGHT};
    mixdQLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ};

    atb::infer::LinearParam mixdKLinearParam;
    mixdKLinearParam.transposeA = false;
    mixdKLinearParam.transposeB = false;
    mixdKLinearParam.hasBias = false;
    CreateOp(mixdKLinearParam, &mixdKLinearNode.op);
    mixdKLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_KMIXDWEIGHT};
    mixdKLinearNode.outTensorIds = {INTERMIDATE_MIXEDK};

    atb::infer::LinearParam mixdVLinearParam;
    mixdVLinearParam.transposeA = false;
    mixdVLinearParam.transposeB = false;
    mixdVLinearParam.hasBias = false;
    CreateOp(mixdVLinearParam, &mixdVLinearNode.op);
    mixdVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_VMIXDWEIGHT};
    mixdVLinearNode.outTensorIds = {INTERMIDATE_MIXEDV};

    atb::infer_old::PositionEmbedding1dFusionParam positionEmbedding1dFusionParam;
    positionEmbedding1dFusionParam.headNum = param.headNum;
    positionEmbedding1dFusionParam.rotaryCoeff = param.rotaryCoeff;
    CreateOp(positionEmbedding1dFusionParam, &ropeNode.op);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    atb::infer_old::SelfAttentionKvCacheFusionParam  selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.dk = param.dk;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.layerId = param.layerId;
    selfAttentionKvCacheParam.tokenOffset = param.tokenOffset;
    selfAttentionKvCacheParam.seqLen = param.seqLen;
    CreateOp(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.op);
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_MIXEDV,
                                            IN_CACHEK,
                                            IN_CACHEV,
                                            INTERMIDATE_POSITIONEMBEDQ,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[1] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(4) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[1] / param.headNum;
    };

    atb::infer_old::LinearParallelParam selfOutLinearParallelParam;
    selfOutLinearParallelParam.transWeight = false;
    selfOutLinearParallelParam.rank = param.rank;
    selfOutLinearParallelParam.rankSize = param.rankSize;
    selfOutLinearParallelParam.rankRoot = 0;
    selfOutLinearParallelParam.bias = "None";
    selfOutLinearParallelParam.parallelType = "RowParallel";
    selfOutLinearParallelParam.backend = "hccl";
    CreateOp(selfOutLinearParallelParam, &selfOutLinearParallelNode.op);
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    atb::infer_old::AddParam selfResidualAddParam;
    CreateOp(selfResidualAddParam, &selfResidualAddNode.op);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
    selfResidualAddNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = oldShape.dims[2];
    };

    atb::infer_old::RmsNormParam selfNormParam;
    selfNormParam.rmsNormEps = param.rmsNormEps;
    CreateOp(selfNormParam, &selfNormNode.op);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb::infer_old::MlpParam mlpParam;
    mlpParam.model = "llama13b";
    CreateOp(mlpParam, &mlpNode.op);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEWEIGHT, IN_MLPUPWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    atb::infer_old::LinearParallelParam mlpLinearParallelParam;
    mlpLinearParallelParam.transWeight = false;
    mlpLinearParallelParam.rank = param.rank;
    mlpLinearParallelParam.rankSize = param.rankSize;
    mlpLinearParallelParam.rankRoot = 0;
    mlpLinearParallelParam.bias = "None";
    mlpLinearParallelParam.parallelType = "RowParallel";
    mlpLinearParallelParam.backend = "hccl";
    CreateOp(mlpLinearParallelParam, &mlpLinearParallelNode.op);
    mlpLinearParallelNode.inTensorIds = {INTERMIDATE_MLPOUT, IN_MLPDOWNWEIGHT};
    mlpLinearParallelNode.outTensorIds = {INTERMIDATE_MLPLINEARPARALLELOUT};

    atb::infer_old::AddParam mlpResidualAddParam;
    CreateOp(mlpResidualAddParam, &mlpResidualAddNode.op);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPLINEARPARALLELOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA13BLAYEROUT};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOp(opGraph, operation);
#endif
    return atb::NO_ERROR;
}

LlamaLayerFusionParallelBinder::LlamaLayerFusionParallelBinder() {}

LlamaLayerFusionParallelBinder::~LlamaLayerFusionParallelBinder() {}

void LlamaLayerFusionParallelBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }

    layerId_ = paramJson["layerId"].get<int>();
}

void LlamaLayerFusionParallelBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = 16;
    const uint32_t seqLenTensorId = 17;
    const uint32_t layerIdTensorId = 18;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;
}
} // namespace atb_speed