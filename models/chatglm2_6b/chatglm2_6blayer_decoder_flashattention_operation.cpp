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
#include "chatglm2_6blayer_decoder_flashattention_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/mlp_operation.h"

namespace AclTransformer {
enum Chatglm2LayerDecoderTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPLINEARWEIGHTUP,
    IN_MLPLINEARWEIGHTDOWN,
    IN_ROPECACHE,
    IN_PRESENTKEY,
    IN_PRESENTVALUE,
    IN_ATTENTIONMASK,
    IN_SEQLEN,
    IN_TOKENOFFSET,
    IN_LAYERID,
    OUT_GLMLAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_PRESENTKEY,
    INTERMIDATE_PRESENTVALUE,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 15;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 9;

ChatGlm2LayerDecoderFlashAttentionOperation::ChatGlm2LayerDecoderFlashAttentionOperation(
    const ChatGlm2LayerDecoderFlashAttentionParam &param)
    : GraphOperation("ChatGlm2LayerDecoderFlashAttentionOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdQkvLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS};
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    positionEmbeddingNode.operation.reset(new AclTransformer::PositionEmbeddingOperation({true, param_.numHeadsPerPartition,
        param_.numHeadsPerPartition, param_.hiddenSizePerHead, param_.numGroupsPerPartition, param_.hiddenSizePerHead, 0, param_.model}));
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_ROPECACHE};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_PRESENTKEY, INTERMIDATE_PRESENTVALUE};

    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param_.headNum;
    selfAttentionKvCacheParam.layerId = param_.layerId;
    selfAttentionKvCacheParam.dk = param_.dk;
    selfAttentionKvCacheParam.numHeadsPerPartition = param_.numHeadsPerPartition;
    selfAttentionKvCacheParam.hiddenSizePerHead = param_.hiddenSizePerHead;
    selfAttentionKvCacheParam.numGroupsPerPartition = param_.numGroupsPerPartition;
    selfAttentionKvCacheParam.tokenOffset = param_.tokenOffset;
    selfAttentionKvCacheParam.seqLen = param_.seqLen;
    selfAttentionKvCacheParam.model = param_.model;
    selfAttentionKvCacheNode.operation.reset(new AclTransformer::SelfAttentionKvCacheFusionOperation(selfAttentionKvCacheParam));
    selfAttentionKvCacheNode.inTensorIds = {
                                            INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_PRESENTKEY,
                                            INTERMIDATE_PRESENTVALUE,
                                            IN_PRESENTKEY,
                                            IN_PRESENTVALUE,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.useVariantPackParam = true;

    AclTransformer::LinearParam selfOutlinearParam;
    selfOutlinearParam.hasBias = false;

    selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation(selfOutlinearParam));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    selfNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    AclTransformer::MlpParam mlpParam;
    mlpParam.model = param_.model;
    mlpNode.operation.reset(new AclTransformer::MlpOperation(mlpParam));
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPLINEARWEIGHTUP, IN_MLPLINEARWEIGHTDOWN};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_GLMLAYEROUT};
}

ChatGlm2LayerDecoderFlashAttentionOperation::~ChatGlm2LayerDecoderFlashAttentionOperation() {}

uint64_t ChatGlm2LayerDecoderFlashAttentionOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm2LayerDecoderFlashAttentionOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status ChatGlm2LayerDecoderFlashAttentionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
