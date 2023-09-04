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
#include "chatglm2_6b_quant_layer_encoder_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/rms_norm_quant_operation.h"
#include "acltransformer/ops/quant_operation.h"
#include "acltransformer/ops/linear_quant_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/mlp_quant_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "iostream"

namespace AclTransformer {
enum Chatglm6BQuantLayerEncoderTensorId {
    IN_HIDDENSTATES = 0,

    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDDEQSCALE,
    IN_QKVMIXDBIAS,

    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARDEQSCALE,
    IN_SELFOUTLINEARBIAS,

    IN_MLPLINEARWEIGHTUP,
    IN_MLPLINEARDEQSCALETUP,
    IN_MLPLINEARBIASTUP,

    IN_MLPLINEARWEIGHTDOWN,
    IN_MLPLINEARDEQSCALEDOWN,
    IN_MLPLINEARBIASDOWN,

    IN_NORMWEIGHT,
    IN_SELFOUTNORMWEIGHT,

    IN_ROPECACHE,
    IN_ATTENTIONMASK,
    IN_BETA,
    
    OUT_GLMLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFQUANTOUT,

    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 18;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 9;
static const uint64_t NODE_COUNT = 10;

ChatGlm2QuantLayerEncoderOperation::ChatGlm2QuantLayerEncoderOperation(const ChatGlm2QuantLayerParam &param)
    : GraphOperation("ChatGlm2QuantLayerEncoderOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdQkvLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutQuantNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::RmsNormQuantOperation({param_.qkvInputScale, param_.qkvInputOffset}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_BETA};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdQkvLinearNode.operation.reset(new AclTransformer::LinearQuantOperation({}));
    mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS, IN_QKVMIXDDEQSCALE};
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    positionEmbeddingNode.operation.reset(new AclTransformer::PositionEmbeddingOperation({true, param_.numHeadsPerPartition,
        param_.numHeadsPerPartition, param_.hiddenSizePerHead, param_.numGroupsPerPartition, param_.hiddenSizePerHead, 0, param_.model}));
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_ROPECACHE};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.transKey = param_.transKey;
    selfAttentionParam.dk = param_.hiddenSizePerHead;
    selfAttentionParam.headNum = param_.numHeadsPerPartition;
    selfAttentionParam.layerId = param_.layerId;
    selfAttentionParam.preScale = param_.preScale;
    selfAttentionParam.postScale = param_.postScale;
    selfAttentionParam.numHeadsPerPartition = param_.numHeadsPerPartition;
    selfAttentionParam.hiddenSizePerHead = param_.hiddenSizePerHead;
    selfAttentionParam.numGroupsPerPartition = param_.numGroupsPerPartition;
    selfAttentionParam.model = param_.model;

    selfAttentionNode.operation.reset(new AclTransformer::SelfAttentionOperation(selfAttentionParam));
    selfAttentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, OUT_PRESENTKEY, OUT_PRESENTVALUE, IN_ATTENTIONMASK};
    selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT};

    selfOutQuantNode.operation.reset(new AclTransformer::QuantOperation({param_.denseInputScale, param_.denseInputOffset}));
    selfOutQuantNode.inTensorIds = {INTERMIDATE_SELFOUT};
    selfOutQuantNode.outTensorIds = {INTERMIDATE_SELFQUANTOUT};

    selfOutLinearNode.operation.reset(new AclTransformer::LinearQuantOperation({}));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFQUANTOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS, IN_SELFOUTLINEARDEQSCALE};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    selfNormNode.operation.reset(new AclTransformer::RmsNormQuantOperation({param_.selfLnInputScale, param_.selfLnInputOffset}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_BETA};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    AclTransformer::MlpQuantParam mlpQuantParam;
    mlpQuantParam.inputScale = param_.ffnOutInputScale;
    mlpQuantParam.inputOffset = param_.ffnOutInputOffset;
    mlpQuantParam.model = "chatglm2_6b";
    mlpNode.operation.reset(new AclTransformer::MlpQuantOperation(mlpQuantParam));
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPLINEARWEIGHTUP, IN_MLPLINEARBIASTUP, IN_MLPLINEARDEQSCALETUP, IN_MLPLINEARWEIGHTDOWN, IN_MLPLINEARBIASDOWN, IN_MLPLINEARDEQSCALEDOWN};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_GLMLAYEROUT};
}

ChatGlm2QuantLayerEncoderOperation::~ChatGlm2QuantLayerEncoderOperation() {}

uint64_t ChatGlm2QuantLayerEncoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm2QuantLayerEncoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status ChatGlm2QuantLayerEncoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    const size_t glmLayerOutID = 0;
    const size_t presentKeyID = 1;
    const size_t presentValueID = 2;
    outTensorDescs.at(glmLayerOutID) = inTensors.at(IN_HIDDENSTATES).desc;
    outTensorDescs.at(presentKeyID) = inTensors.at(IN_HIDDENSTATES).desc;

    outTensorDescs.at(presentKeyID).dims.at(2) = param_.numGroupsPerPartition;
    outTensorDescs.at(presentKeyID).dims.push_back(param_.hiddenSizePerHead);

    outTensorDescs.at(presentValueID) = outTensorDescs.at(presentKeyID);
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer