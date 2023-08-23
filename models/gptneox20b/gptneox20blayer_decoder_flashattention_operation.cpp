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
#include "gptneox20blayer_decoder_flashattention_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
enum GptNeox20BLayerDecoderFlashAttentionTensorId {
    IN_HIDDENSTATES = 0,
    IN_INPUTLAYERNORMWEIGTH,
    IN_INPUTLAYERNORMBIAS,
    IN_POSTATTNLAYERNORMWEIGHT,
    IN_POSTATTNLAYERNORMBIAS,
    IN_QKVWEIGHT,
    IN_QKVBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_FFNLINEARWEIGHT,
    IN_FFNLINEARBIAS,
    IN_FFNOUTLINEARWEIGHT,
    IN_FFNOUTLINEARBIAS,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    OUT_GPTNEOXLAYEROUT,
    INTERMEDIATE_INPUTLAYERNORMOUT,
    INTERMEDIATE_MIXEDQKVLINEAROUT,
    INTERMEDIATE_QUERYEMBED,
    INTERMEDIATE_KEYEMBED,
    INTERMEDIATE_VALUE,
    INTERMEDIATE_SELFATTNOUT,
    INTERMEDIATE_SELFATTNLINEAROUT,
    INTERMEDIATE_POSTATTNLAYERNORMOUT,
    INTERMEDIATE_FFNLINEAROUT,
    INTERMEDIATE_FFNOUTLINEAROUT,
    INTERMEDIATE_ATTNRESIDUALADDOUT,   
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

GptNeox20BLayerDecoderFlashAttentionOperation::GptNeox20BLayerDecoderFlashAttentionOperation(
    const GptNeox20BLayerDecoderFlashAttentionParam &param)
    : GraphOperation("GptNeox20BLayerDecoderFlashAttentionOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputLayerNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &qkvLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionKvCacheFusionNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttnLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &postAttnLayerNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &attnResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnResidualAddNode = opGraph_.nodes.at(nodeId++);
    
    // params
    AclTransformer::NormParam inputLayerNormParam;
    inputLayerNormParam.layerNormEps = param_.layerNormEps;
    inputLayerNormParam.beginNormAxis = 2;
    inputLayerNormParam.beginParamsAxis = 2;
    
    AclTransformer::LinearParam qkvLinearParam;
    qkvLinearParam.transposeA = false;
    qkvLinearParam.transposeB = false;
    
    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = param_.headNum;
    positionEmbeddingParam.model = "gptneox20b";
    positionEmbeddingParam.dk = param_.dk;
    positionEmbeddingParam.rotaryPct = param_.rotaryPct;
    
    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheFusionParam;
    selfAttentionKvCacheFusionParam.headNum = param_.headNum;
    selfAttentionKvCacheFusionParam.layerId = 0;
    selfAttentionKvCacheFusionParam.dk = param_.dk;
    selfAttentionKvCacheFusionParam.tokenOffset = param_.tokenOffset;
    selfAttentionKvCacheFusionParam.seqLen = param_.seqLen;
    
    AclTransformer::LinearParam selfAttnLinearParam;
    selfAttnLinearParam = qkvLinearParam;
    
    AclTransformer::NormParam postAttnLayerNormParam;
    postAttnLayerNormParam = inputLayerNormParam;
    
    AclTransformer::FfnParam ffnLinearParam;
    ffnLinearParam.activationFuncType = AclTransformer::FfnParam::ActivationFuncType::GELU;
    
    AclTransformer::LinearParam ffnOutLinearParam;
    ffnOutLinearParam = qkvLinearParam;
    
    AclTransformer::AddParam ffnResidualAddParam;
    
    AclTransformer::AddParam attnResidualAddParam;

    inputLayerNormNode.operation.reset(new AclTransformer::NormOperation(inputLayerNormParam));
    inputLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_INPUTLAYERNORMWEIGTH, IN_INPUTLAYERNORMBIAS};
    inputLayerNormNode.outTensorIds = {INTERMEDIATE_INPUTLAYERNORMOUT};

    qkvLinearNode.operation.reset(new AclTransformer::LinearOperation(qkvLinearParam));
    qkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTLAYERNORMOUT, IN_QKVWEIGHT, IN_QKVBIAS};
    qkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDQKVLINEAROUT};

    positionEmbeddingNode.operation.reset(new AclTransformer::PositionEmbeddingOperation(positionEmbeddingParam));
    positionEmbeddingNode.inTensorIds = {INTERMEDIATE_MIXEDQKVLINEAROUT, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
    positionEmbeddingNode.outTensorIds = {INTERMEDIATE_QUERYEMBED, INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE};

    selfAttentionKvCacheFusionNode.operation.reset(
        new AclTransformer::SelfAttentionKvCacheFusionOperation(selfAttentionKvCacheFusionParam));
    selfAttentionKvCacheFusionNode.inTensorIds = {
        INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE, IN_CACHEK, IN_CACHEV, INTERMEDIATE_QUERYEMBED, IN_ATTENTIONMASK,
        IN_TOKENOFFSET, IN_SEQLEN, IN_LAYERID};
    selfAttentionKvCacheFusionNode.outTensorIds = {INTERMEDIATE_SELFATTNOUT};
    selfAttentionKvCacheFusionNode.useVariantPackParam = true;

    selfAttnLinearNode.operation.reset(new AclTransformer::LinearOperation(selfAttnLinearParam));
    selfAttnLinearNode.inTensorIds = {INTERMEDIATE_SELFATTNOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfAttnLinearNode.outTensorIds = {INTERMEDIATE_SELFATTNLINEAROUT};
    
    postAttnLayerNormNode.operation.reset(new AclTransformer::NormOperation(postAttnLayerNormParam));
    postAttnLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_POSTATTNLAYERNORMWEIGHT, IN_POSTATTNLAYERNORMBIAS};
    postAttnLayerNormNode.outTensorIds = {INTERMEDIATE_POSTATTNLAYERNORMOUT};
    
    ffnLinearNode.operation.reset(new AclTransformer::FfnOperation(ffnLinearParam));
    ffnLinearNode.inTensorIds = {INTERMEDIATE_POSTATTNLAYERNORMOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS};
    ffnLinearNode.outTensorIds = {INTERMEDIATE_FFNLINEAROUT};
    
    ffnOutLinearNode.operation.reset(new AclTransformer::LinearOperation(ffnOutLinearParam));
    ffnOutLinearNode.inTensorIds = {INTERMEDIATE_FFNLINEAROUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS};
    ffnOutLinearNode.outTensorIds = {INTERMEDIATE_FFNOUTLINEAROUT};

    attnResidualAddNode.operation.reset(new AclTransformer::AddOperation(attnResidualAddParam));
    attnResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMEDIATE_SELFATTNLINEAROUT};
    attnResidualAddNode.outTensorIds = {INTERMEDIATE_ATTNRESIDUALADDOUT};

    ffnResidualAddNode.operation.reset(new AclTransformer::AddOperation(ffnResidualAddParam));
    ffnResidualAddNode.inTensorIds = {INTERMEDIATE_ATTNRESIDUALADDOUT, INTERMEDIATE_FFNOUTLINEAROUT};
    ffnResidualAddNode.outTensorIds = {OUT_GPTNEOXLAYEROUT};
}

GptNeox20BLayerDecoderFlashAttentionOperation::~GptNeox20BLayerDecoderFlashAttentionOperation() {}

uint64_t GptNeox20BLayerDecoderFlashAttentionOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t GptNeox20BLayerDecoderFlashAttentionOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status GptNeox20BLayerDecoderFlashAttentionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;  // [bs, sq, hn * hs]
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer