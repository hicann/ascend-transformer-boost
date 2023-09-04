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
#include "chatglm2_6b_quant_layer_decoder_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/rms_norm_quant_operation.h"
#include "acltransformer/ops/quant_operation.h"
#include "acltransformer/ops/linear_quant_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/mlp_quant_operation.h"
#include "acltransformer/ops/mlp_operation.h"

namespace AclTransformer {
enum Chatglm2QuantLayerDecoderTensorId {
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
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_BETA,

    OUT_GLMLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_VALUE,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFQUANTOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

ChatGlm2QuantLayerDecoderOperation::ChatGlm2QuantLayerDecoderOperation(const ChatGlm2QuantLayerParam &param)
    : GraphOperation("ChatGlm2QuantLayerDecoderOperation"), param_(param)
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
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE};

    AclTransformer::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.transKey = param_.transKey;
    selfAttentionKvCacheParam.dk = param_.hiddenSizePerHead;
    selfAttentionKvCacheParam.headNum = param_.numHeadsPerPartition;
    selfAttentionKvCacheParam.layerId = param_.layerId;
    selfAttentionKvCacheParam.preScale = param_.preScale;
    selfAttentionKvCacheParam.postScale = param_.postScale;
    selfAttentionKvCacheParam.numHeadsPerPartition = param_.numHeadsPerPartition;
    selfAttentionKvCacheParam.hiddenSizePerHead = param_.hiddenSizePerHead;
    selfAttentionKvCacheParam.numGroupsPerPartition = param_.numGroupsPerPartition;
    selfAttentionKvCacheParam.model = param_.model;

    selfAttentionKvCacheNode.operation.reset(new AclTransformer::SelfAttentionKvCacheOperation(selfAttentionKvCacheParam));
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_VALUE,
                                            IN_PASTKEY,
                                            IN_PASTVALUE};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};

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

ChatGlm2QuantLayerDecoderOperation::~ChatGlm2QuantLayerDecoderOperation() {}

uint64_t ChatGlm2QuantLayerDecoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm2QuantLayerDecoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status ChatGlm2QuantLayerDecoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    const AsdOps::Tensor &hiddenStateTensor = inTensors.at(IN_HIDDENSTATES);
    const AsdOps::Tensor &keyTensor = inTensors.at(IN_PASTKEY);
    const AsdOps::Tensor &valueTensor = inTensors.at(IN_PASTVALUE);
    const size_t glmLayerOutID = 0;
    const size_t presentKeyID = 1;
    const size_t presentValueID = 2;

    outTensorDescs.at(glmLayerOutID) = hiddenStateTensor.desc;

    outTensorDescs.at(presentKeyID) = keyTensor.desc;
    outTensorDescs.at(presentKeyID).dims.at(0) += 1;

    outTensorDescs.at(presentValueID) = valueTensor.desc;
    outTensorDescs.at(presentValueID).dims.at(0) += 1;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer