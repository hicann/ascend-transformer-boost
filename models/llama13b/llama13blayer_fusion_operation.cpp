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
#include "llama13blayer_fusion_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/position_embedding_1d_fusion_operation.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
enum LLaMA13BLayerTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_MIXEDQKVLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
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
    INTERMIDATE_MIXEDQKVLINEAROUT,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 16;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 9;

LLaMA13BLayerFusionOperation::LLaMA13BLayerFusionOperation(const LLaMA13BLayerFusionParam &param)
    : GraphOperation("LLaMA13BLayerFusionOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdQKVLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ropeNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    AclTransformer::LinearParam linearParam;
    linearParam.hasBias = false;

    inputNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdQKVLinearNode.operation.reset(new AclTransformer::LinearOperation(linearParam));
    mixdQKVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_MIXEDQKVLINEARWEIGHT};
    mixdQKVLinearNode.outTensorIds = {INTERMIDATE_MIXEDQKVLINEAROUT};

    AclTransformer::PositionEmbedding1dFusionParam positionEmbedding1dFusionParam;
    positionEmbedding1dFusionParam.headNum = param_.headNum;
    positionEmbedding1dFusionParam.rotaryCoeff = param_.rotaryCoeff;
    positionEmbedding1dFusionParam.model = param_.model;
    ropeNode.operation.reset(new AclTransformer::PositionEmbedding1dSplitFusionOperation(positionEmbedding1dFusionParam));
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQKVLINEAROUT, IN_COSTABLE, IN_SINTABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV};
    ropeNode.inTensorViewFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        oriLinearDim = oldDims;
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param_.headNum;
    selfAttentionKvCacheParam.layerId = param_.layerId;
    selfAttentionKvCacheParam.dk = param_.dk;
    selfAttentionKvCacheParam.tokenOffset = param_.tokenOffset;
    selfAttentionKvCacheParam.seqLen = param_.seqLen;
    selfAttentionKvCacheParam.model = param_.model;
    selfAttentionKvCacheNode.operation.reset(
        new AclTransformer::SelfAttentionKvCacheFusionOperation(selfAttentionKvCacheParam));
    
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
    selfAttentionKvCacheNode.useVariantPackParam = true;
    selfAttentionKvCacheNode.inTensorViewFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oriLinearDim.at(0), oriLinearDim.at(1), param_.headNum, oldDims.at(1) / param_.headNum};
    };
    selfAttentionKvCacheNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oriLinearDim.at(0), oriLinearDim.at(1), param_.headNum, oldDims.at(1) / param_.headNum};
    };
    selfAttentionKvCacheNode.inTensorViewFuncs.at(4) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oriLinearDim.at(0), oriLinearDim.at(1), param_.headNum, oldDims.at(1) / param_.headNum};
    };

    selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation(linearParam));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    selfNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    mlpNode.operation.reset(new AclTransformer::MlpOperation({}));
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT, IN_MLPUPWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA13BLAYEROUT};
}

LLaMA13BLayerFusionOperation::~LLaMA13BLayerFusionOperation() {}

uint64_t LLaMA13BLayerFusionOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t LLaMA13BLayerFusionOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status LLaMA13BLayerFusionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                     AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer