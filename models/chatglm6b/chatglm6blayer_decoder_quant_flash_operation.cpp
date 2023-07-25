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
#include "chatglm6blayer_decoder_quant_flash_operation.h"
#include "acltransformer/ops/quant_operation.h"
#include "acltransformer/ops/norm_quant_operation.h"
#include "acltransformer/ops/add_norm_quant_operation.h"
#include "acltransformer/ops/linear_quant_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_quant_operation.h"

namespace AclTransformer {
enum Chatglm6BLayerDecoderQuantFlashTensorId {
    IN_HIDDENSTATES = 0,
    IN_QKVMIXDWEIGHT,
    IN_QKVDEQSCALE,
    IN_QKVMIXDBIAS,
    IN_DENSEQUANTWEIGHT,
    IN_DENSEQUANTDEQSCALE,
    IN_DENSEQUANTBIAS,
    IN_FFNLINEARWEIGHT,
    IN_FFNLINEARDEQSCALE,
    IN_FFNLINEARBIAS,
    IN_FFNOUTLINEARWEIGHT,
    IN_FFNOUTLINEARDEQSCALE,
    IN_FFNOUTLINEARBIAS,
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_POSTNORMWEIGHT,
    IN_POSTNORMBIAS,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_SEQLEN,
    IN_TOKENOFFSET,
    IN_LAYERID,
    IN_RES,

    OUT_GLMLAYEROUT,
    OUT_SELFNORMRES,

    INTERMIDATE_INPUTNORMQUANT,
    INTERMIDATE_INPUTNORMRES,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_VALUE,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_DENSEQUANTOUT,
    INTERMIDATE_DENSELINEARQUANTOUT,
    INTERMIDATE_SELFNORMQUANTOUT,
    INTERMIDATE_FFNOUT,
    INTERMIDATE_FFNQUANTOUT,
};

static const uint64_t IN_TENSOR_COUNT = 27;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 10;

ChatGlm6BLayerDecoderQuantFlashOperation::ChatGlm6BLayerDecoderQuantFlashOperation(
    const ChatGlm6BLayerQuantFlashParam &param)
    : GraphOperation("ChatGlm6BLayerDecoderQuantFlashOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputQuantNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdQkvLinearQuantNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &denseQuantNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &denseLinearQuantNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfLayernormQuantNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnLinearQuantNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnOutQuantNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnOutLinearQuantNode = opGraph_.nodes.at(nodeId++);

    inputQuantNode.operation.reset(new AclTransformer::AddNormQuantOperation(
        {param_.layerNormEps, param_.qkvInputScale, param_.qkvInputOffset, param_.residualAddScale}));
    inputQuantNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS, IN_RES};
    inputQuantNode.outTensorIds = {INTERMIDATE_INPUTNORMQUANT, INTERMIDATE_INPUTNORMRES};

    mixdQkvLinearQuantNode.operation.reset(new AclTransformer::LinearQuantOperation({}));
    mixdQkvLinearQuantNode.inTensorIds = {INTERMIDATE_INPUTNORMQUANT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS, IN_QKVDEQSCALE};
    mixdQkvLinearQuantNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    positionEmbeddingNode.operation.reset(new AclTransformer::RopeOperation({param_.headNum}));
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE,
                                         IN_SEQLEN};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE};

    // add flash attention
    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param_.headNum;
    selfAttentionKvCacheParam.layerId = param_.layerId;
    selfAttentionKvCacheParam.dk = param_.dk;
    selfAttentionKvCacheParam.tokenOffset = param_.tokenOffset;
    selfAttentionKvCacheParam.seqLen = param_.seqLen;
    selfAttentionKvCacheNode.operation.reset(
        new AclTransformer::SelfAttentionKvCacheFusionOperation(selfAttentionKvCacheParam));
    selfAttentionKvCacheNode.inTensorIds = {
        INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE, IN_CACHEK,      IN_CACHEV, INTERMIDATE_POSITIONEMBEDQ,
        IN_ATTENTIONMASK,           IN_SEQLEN,         IN_TOKENOFFSET, IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.useVariantPackParam = true;

    denseQuantNode.operation.reset(
        new AclTransformer::QuantOperation({param_.denseInputScale, param_.denseInputOffset}));
    denseQuantNode.inTensorIds = {INTERMIDATE_SELFOUT};
    denseQuantNode.outTensorIds = {INTERMIDATE_DENSEQUANTOUT};

    denseLinearQuantNode.operation.reset(new AclTransformer::LinearQuantOperation({}));
    denseLinearQuantNode.inTensorIds = {INTERMIDATE_DENSEQUANTOUT, IN_DENSEQUANTWEIGHT, IN_DENSEQUANTBIAS,
                                        IN_DENSEQUANTDEQSCALE};
    denseLinearQuantNode.outTensorIds = {INTERMIDATE_DENSELINEARQUANTOUT};

    selfLayernormQuantNode.operation.reset(new AclTransformer::AddNormQuantOperation(
        {param_.layerNormEps, param_.selfLnInputScale, param_.selfLnInputOffset, param_.residualAddScale}));
    selfLayernormQuantNode.inTensorIds = {INTERMIDATE_DENSELINEARQUANTOUT, IN_POSTNORMWEIGHT, IN_POSTNORMBIAS,
                                          INTERMIDATE_INPUTNORMRES};
    selfLayernormQuantNode.outTensorIds = {INTERMIDATE_SELFNORMQUANTOUT, OUT_SELFNORMRES};

    ffnLinearQuantNode.operation.reset(new AclTransformer::FfnQuantOperation({}));
    ffnLinearQuantNode.inTensorIds = {INTERMIDATE_SELFNORMQUANTOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS,
                                      IN_FFNLINEARDEQSCALE};
    ffnLinearQuantNode.outTensorIds = {INTERMIDATE_FFNOUT};

    ffnOutQuantNode.operation.reset(
        new AclTransformer::QuantOperation({param_.ffnOutInputScale, param_.ffnOutInputOffset}));
    ffnOutQuantNode.inTensorIds = {INTERMIDATE_FFNOUT};
    ffnOutQuantNode.outTensorIds = {INTERMIDATE_FFNQUANTOUT};

    ffnOutLinearQuantNode.operation.reset(new AclTransformer::LinearQuantOperation({}));
    ffnOutLinearQuantNode.inTensorIds = {INTERMIDATE_FFNQUANTOUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS,
                                         IN_FFNOUTLINEARDEQSCALE};
    ffnOutLinearQuantNode.outTensorIds = {OUT_GLMLAYEROUT};
}

ChatGlm6BLayerDecoderQuantFlashOperation::~ChatGlm6BLayerDecoderQuantFlashOperation() {}

uint64_t ChatGlm6BLayerDecoderQuantFlashOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm6BLayerDecoderQuantFlashOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status
ChatGlm6BLayerDecoderQuantFlashOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                         AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer