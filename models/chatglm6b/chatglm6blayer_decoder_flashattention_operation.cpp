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
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/position_embedding_fusion_rope_operation.h"

namespace AclTransformer {
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
    IN_COSSUM_ID,
    IN_SINSUM_ID,
    IN_ATTENTIONMASK_ID,
    IN_CACHEK_ID,
    IN_CACHEV_ID,
    IN_SEQLEN_ID,
    IN_TOKENOFFSET_ID,
    IN_LAYERID_ID,
    OUT_LAYEROUT_ID,
    INTERMEDIATE_INPUTNORMOUT_ID,
    INTERMEDIATE_MIXEDLINEAROUTQKV_ID,
    INTERMEDIATE_POSITIONEMBEDQ_ID,
    INTERMEDIATE_POSITIONEMBEDK_ID,
    INTERMEDIATE_VALUE_ID,
    INTERMEDIATE_SELFOUT_ID,
    INTERMEDIATE_SELFLINEAROUT_ID,
    INTERMEDIATE_SELFRESIDUALADDOUT_ID,
    INTERMEDIATE_SELFNORMOUT_ID,
    INTERMEDIATE_FFNOUT,
    INTERMEDIATE_FFNLINEAROUT_ID,
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

ChatGlm6BLayerDecoderFlashAttentionOperation::ChatGlm6BLayerDecoderFlashAttentionOperation(
    const ChatGlm6BLayerDecoderFlashAttentionParam &param)
    : GraphOperation("ChatGlm6BLayerDecoderFlashAttentionOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &inputNormNode = opGraph_.nodes.at(nodeId++);
    auto &mixdQkvLinearNode = opGraph_.nodes.at(nodeId++);
    auto &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    auto &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
    auto &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    auto &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    auto &selfNormNode = opGraph_.nodes.at(nodeId++);
    auto &ffnNode = opGraph_.nodes.at(nodeId++);
    auto &ffnLinearNode = opGraph_.nodes.at(nodeId++);
    auto &ffnResidualAddNode = opGraph_.nodes.at(nodeId++);

    AclTransformer::NormParam inputNormParam;
    inputNormParam.layerNormEps = param_.layerNormEps;
    inputNormNode.operation.reset(new AclTransformer::NormOperation(inputNormParam));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_NORMWEIGHT_ID, IN_NORMBIAS_ID};
    inputNormNode.outTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID};

    mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation(AclTransformer::LinearParam()));
    mixdQkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID, IN_QKVMIXEDWEIGHT_ID, IN_QKVMIXEDBIAS_ID};
    mixdQkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID};

    positionEmbeddingNode.operation.reset(new AclTransformer::OptRopeOperation({param_.headNum}));
    positionEmbeddingNode.inTensorIds = {INTERMEDIATE_MIXEDLINEAROUTQKV_ID, IN_COSSUM_ID,
                                         IN_SINSUM_ID, IN_SEQLEN_ID};
    positionEmbeddingNode.outTensorIds = {INTERMEDIATE_POSITIONEMBEDQ_ID, INTERMEDIATE_POSITIONEMBEDK_ID,
                                          INTERMEDIATE_VALUE_ID};

    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param_.headNum;
    selfAttentionKvCacheParam.layerId = param_.layerId;
    selfAttentionKvCacheParam.dk = param_.dk;
    selfAttentionKvCacheParam.tokenOffset = param_.tokenOffset;
    selfAttentionKvCacheParam.seqLen = param_.seqLen;
    selfAttentionKvCacheNode.operation.reset(
        new AclTransformer::SelfAttentionKvCacheFusionOperation(selfAttentionKvCacheParam));
    selfAttentionKvCacheNode.inTensorIds = {INTERMEDIATE_POSITIONEMBEDK_ID,
                                            INTERMEDIATE_VALUE_ID,
                                            IN_CACHEK_ID,
                                            IN_CACHEV_ID,
                                            INTERMEDIATE_POSITIONEMBEDQ_ID,
                                            IN_ATTENTIONMASK_ID,
                                            IN_SEQLEN_ID,
                                            IN_TOKENOFFSET_ID,
                                            IN_LAYERID_ID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMEDIATE_SELFOUT_ID};
    selfAttentionKvCacheNode.useVariantPackParam = true;

    selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation(AclTransformer::LinearParam()));
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFOUT_ID, IN_SELFOUTLINEARWEIGHT_ID, IN_SELFOUTLINEARBIAS_ID};
    selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT_ID};

    AclTransformer::AddParam selfResidualAddParam;
    selfResidualAddParam.scale = param_.residualAddScale;
    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation(selfResidualAddParam));
    selfResidualAddNode.inTensorIds = {INTERMEDIATE_INPUTNORMOUT_ID, INTERMEDIATE_SELFLINEAROUT_ID};
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_SELFRESIDUALADDOUT_ID};

    AclTransformer::NormParam selfNormParam;
    selfNormParam.layerNormEps = param_.layerNormEps;
    selfNormNode.operation.reset(new AclTransformer::NormOperation(selfNormParam));
    selfNormNode.inTensorIds = {INTERMEDIATE_SELFRESIDUALADDOUT_ID, IN_SELFOUTNORMWEIGHT_ID, IN_SELFOUTNORMBIAS_ID};
    selfNormNode.outTensorIds = {INTERMEDIATE_SELFNORMOUT_ID};

    ffnNode.operation.reset(new AclTransformer::FfnOperation(AclTransformer::FfnParam()));
    ffnNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID, IN_FFNLINEARWEIGHT_ID, IN_FFNLINEARBIAS_ID};
    ffnNode.outTensorIds = {INTERMEDIATE_FFNOUT};

    ffnLinearNode.operation.reset(new AclTransformer::LinearOperation(AclTransformer::LinearParam()));
    ffnLinearNode.inTensorIds = {INTERMEDIATE_FFNOUT, IN_FFNOUTLINEARWEIGHT_ID, IN_FFNOUTLINEARBIAS_ID};
    ffnLinearNode.outTensorIds = {INTERMEDIATE_FFNLINEAROUT_ID};

    AclTransformer::AddParam ffnResidualAddParam;
    ffnResidualAddParam.scale = param_.residualAddScale;
    ffnResidualAddNode.operation.reset(new AclTransformer::AddOperation(ffnResidualAddParam));
    ffnResidualAddNode.inTensorIds = {INTERMEDIATE_SELFNORMOUT_ID, INTERMEDIATE_FFNLINEAROUT_ID};
    ffnResidualAddNode.outTensorIds = {OUT_LAYEROUT_ID};
}

ChatGlm6BLayerDecoderFlashAttentionOperation::~ChatGlm6BLayerDecoderFlashAttentionOperation() {}

uint64_t ChatGlm6BLayerDecoderFlashAttentionOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm6BLayerDecoderFlashAttentionOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status
ChatGlm6BLayerDecoderFlashAttentionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                             AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer