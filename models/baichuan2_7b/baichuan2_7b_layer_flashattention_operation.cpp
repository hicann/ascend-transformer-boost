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
#include "baichuan2_7b_layer_flashattention_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"

namespace AclTransformer {
enum Baichuan27BLayerFlashAttentionTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXEDLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
    IN_POSITIONIDS,
    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    OUT_BAICHUAN27BLAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVMIXEDLINEAROUT,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

Baichuan27BLayerFlashAttentionOperation::Baichuan27BLayerFlashAttentionOperation(
    const Baichuan27BLayerFlashAttentionParam &param)
    : GraphOperation("Baichuan27BLayerFlashAttentionOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &qkvLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionKvCacheFusionNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);
    
    // params
    AclTransformer::LinearParam linearParam;
    linearParam.transposeB = param_.transposedWeight;
    linearParam.hasBias = false;

    AclTransformer::MlpParam mlpParam;
    mlpParam.transposeB = !param_.transposedWeight;
    
    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheFusionParam;
    selfAttentionKvCacheFusionParam.headNum = param_.headNum;
    selfAttentionKvCacheFusionParam.layerId = 0;
    selfAttentionKvCacheFusionParam.dk = param_.dk;
    selfAttentionKvCacheFusionParam.tokenOffset = param_.tokenOffset;
    selfAttentionKvCacheFusionParam.seqLen = param_.seqLen;
    selfAttentionKvCacheFusionParam.model = param_.model;

    inputNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    qkvLinearNode.operation.reset(new AclTransformer::LinearOperation(linearParam));
    qkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT};
    qkvLinearNode.outTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};

    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.model = param_.model;
    positionEmbeddingParam.headNum = param_.headNum;
    positionEmbeddingNode.operation.reset(new AclTransformer::PositionEmbeddingOperation(positionEmbeddingParam));
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT, IN_POSITIONIDS, IN_COSEMBED, IN_SINEMBED};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV};

    selfAttentionKvCacheFusionNode.operation.reset(
        new AclTransformer::SelfAttentionKvCacheFusionOperation(selfAttentionKvCacheFusionParam));
    selfAttentionKvCacheFusionNode.inTensorIds = {
        INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV, IN_CACHEK, IN_CACHEV, IN_ATTENTIONMASK,
        IN_TOKENOFFSET, IN_SEQLEN, IN_LAYERID};
    selfAttentionKvCacheFusionNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheFusionNode.useVariantPackParam = true;

    selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation(linearParam));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    selfNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    mlpNode.operation.reset(new AclTransformer::MlpOperation(mlpParam));
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT, IN_MLPUPWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_BAICHUAN27BLAYEROUT};
}

Baichuan27BLayerFlashAttentionOperation::~Baichuan27BLayerFlashAttentionOperation() {}

uint64_t Baichuan27BLayerFlashAttentionOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t Baichuan27BLayerFlashAttentionOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status Baichuan27BLayerFlashAttentionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;  // [bs, sq, hn * hs]
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer