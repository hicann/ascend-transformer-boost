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
#include "chatglm6blayer_decoder_dequant_flashattention_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/dequant_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"

namespace AclTransformer {
enum Chatglm6BLayerDecoderDequantTensorId {
    IN_HIDDENSTATES = 0,
    IN_QKVWEIGHT,
    IN_QKVSCALE,
    IN_DENSEWEIGHT,
    IN_DENSESCALE,
    IN_HTOFHWEIGHT,
    IN_HTOFHSCALE,
    IN_FHTOHWEIGHT,
    IN_FHTOHSCALE,
    
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTNORMBIAS,
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

    OUT_GLMLAYEROUT,

    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVMIXDDEQUANTWEIGHT,
    INTERMIDATE_QKVMIXDDEQUANTWEIGHTT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_VALUE,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFLINEARDEQUANTWEIGHT,
    INTERMIDATE_SELFLINEARDEQUANTWEIGHTT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_FFNLINEARDEQUANTWEIGHT,
    INTERMIDATE_FFNLINEARDEQUANTWEIGHTT,
    INTERMIDATE_FFNOUT,
    INTERMIDATE_FFNOUTLINEARDEQUANTWEIGHT,
    INTERMIDATE_FFNOUTLINEARDEQUANTWEIGHTT,
    INTERMIDATE_FFNLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 30;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 19;
static const uint64_t NODE_COUNT = 18;

ChatGlm6BLayerDecoderDequantFlashAttentionOperation::ChatGlm6BLayerDecoderDequantFlashAttentionOperation(const ChatGlm6BLayerDequantFlashAttentionParam &param)
    : GraphOperation("ChatGlm6BLayerDecoderDequantFlashAttentionOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdWeightDequant = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdWeightDequantT = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdQkvLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutWeightDequant = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutWeightDequantT = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnWeightDequant = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnWeightDequantT = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnLinearWeightDequant = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnLinearWeightDequantT = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ffnResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdWeightDequant.operation.reset(new AclTransformer::DequantOperation({}));
    mixdWeightDequant.inTensorIds = {IN_QKVWEIGHT,IN_QKVSCALE};
    mixdWeightDequant.outTensorIds = {INTERMIDATE_QKVMIXDDEQUANTWEIGHT};

    mixdWeightDequantT.operation.reset(new AclTransformer::TransposeOperation({param_.perm}));
    mixdWeightDequantT.inTensorIds = {INTERMIDATE_QKVMIXDDEQUANTWEIGHT};
    mixdWeightDequantT.outTensorIds = {INTERMIDATE_QKVMIXDDEQUANTWEIGHTT};

    mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, INTERMIDATE_QKVMIXDDEQUANTWEIGHTT, IN_QKVMIXDBIAS};
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    positionEmbeddingNode.operation.reset(new AclTransformer::RopeOperation({param_.headNum}));
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE,
                                         IN_TOKENOFFSET};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE};

    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param_.headNum;
    selfAttentionKvCacheParam.layerId = param_.layerId;
    selfAttentionKvCacheParam.dk = param_.dk;
    selfAttentionKvCacheParam.tokenOffset = param_.tokenOffset;
    selfAttentionKvCacheParam.seqLen = param_.seqLen;
    selfAttentionKvCacheNode.operation.reset(
        new AclTransformer::SelfAttentionKvCacheFusionOperation(selfAttentionKvCacheParam));
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_VALUE,
                                            IN_CACHEK,
                                            IN_CACHEV,
                                            INTERMIDATE_POSITIONEMBEDQ,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.useVariantPackParam = true;

    selfOutWeightDequant.operation.reset(new AclTransformer::DequantOperation({}));
    selfOutWeightDequant.inTensorIds = {IN_DENSEWEIGHT,IN_DENSESCALE};
    selfOutWeightDequant.outTensorIds = {INTERMIDATE_SELFLINEARDEQUANTWEIGHT};

    selfOutWeightDequantT.operation.reset(new AclTransformer::TransposeOperation({param_.perm}));
    selfOutWeightDequantT.inTensorIds = {INTERMIDATE_SELFLINEARDEQUANTWEIGHT};
    selfOutWeightDequantT.outTensorIds = {INTERMIDATE_SELFLINEARDEQUANTWEIGHTT};

    selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, INTERMIDATE_SELFLINEARDEQUANTWEIGHTT, IN_SELFOUTLINEARBIAS};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    selfResidualAddNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    selfNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    ffnWeightDequant.operation.reset(new AclTransformer::DequantOperation({}));
    ffnWeightDequant.inTensorIds = {IN_HTOFHWEIGHT,IN_HTOFHSCALE};
    ffnWeightDequant.outTensorIds = {INTERMIDATE_FFNLINEARDEQUANTWEIGHT};

    ffnWeightDequantT.operation.reset(new AclTransformer::TransposeOperation({param_.perm}));
    ffnWeightDequantT.inTensorIds = {INTERMIDATE_FFNLINEARDEQUANTWEIGHT};
    ffnWeightDequantT.outTensorIds = {INTERMIDATE_FFNLINEARDEQUANTWEIGHTT};

    ffnNode.operation.reset(new AclTransformer::FfnOperation({}));
    ffnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, INTERMIDATE_FFNLINEARDEQUANTWEIGHTT, IN_FFNLINEARBIAS};
    ffnNode.outTensorIds = {INTERMIDATE_FFNOUT};

    ffnLinearWeightDequant.operation.reset(new AclTransformer::DequantOperation({}));
    ffnLinearWeightDequant.inTensorIds = {IN_FHTOHWEIGHT,IN_FHTOHSCALE};
    ffnLinearWeightDequant.outTensorIds = {INTERMIDATE_FFNOUTLINEARDEQUANTWEIGHT};

    ffnLinearWeightDequantT.operation.reset(new AclTransformer::TransposeOperation({param_.perm}));
    ffnLinearWeightDequantT.inTensorIds = {INTERMIDATE_FFNOUTLINEARDEQUANTWEIGHT};
    ffnLinearWeightDequantT.outTensorIds = {INTERMIDATE_FFNOUTLINEARDEQUANTWEIGHTT};

    ffnLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    ffnLinearNode.inTensorIds = {INTERMIDATE_FFNOUT, INTERMIDATE_FFNOUTLINEARDEQUANTWEIGHTT, IN_FFNOUTLINEARBIAS};
    ffnLinearNode.outTensorIds = {INTERMIDATE_FFNLINEAROUT};

    ffnResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    ffnResidualAddNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, INTERMIDATE_FFNLINEAROUT};
    ffnResidualAddNode.outTensorIds = {OUT_GLMLAYEROUT};
}

ChatGlm6BLayerDecoderDequantFlashAttentionOperation::~ChatGlm6BLayerDecoderDequantFlashAttentionOperation() {}

uint64_t ChatGlm6BLayerDecoderDequantFlashAttentionOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm6BLayerDecoderDequantFlashAttentionOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status ChatGlm6BLayerDecoderDequantFlashAttentionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
