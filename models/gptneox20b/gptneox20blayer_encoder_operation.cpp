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
#include "gptneox20blayer_encoder_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
enum GptNeox20BLayerEncoderTensorId {
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
    OUT_GPTNEOXLARYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMEDIATE_INPUTLAYERNORMOUT,
    INTERMEDIATE_MIXEDQKVLINEAROUT,
    INTERMEDIATE_QUERYEMBED,
    INTERMEDIATE_SELFATTNOUT,
    INTERMEDIATE_SELFATTNLINEAROUT,
    INTERMEDIATE_POSTATTNLAYERNORMOUT,
    INTERMEDIATE_FFNLINEAROUT,
    INTERMEDIATE_FFNOUTLINEAROUT,
    INTERMEDIATE_ATTNRESIDUALADDOUT,   
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 9;
static const uint64_t NODE_COUNT = 10;

GptNeox20BLayerEncoderOperation::GptNeox20BLayerEncoderOperation(const GptNeox20BLayerParam &param)
    : GraphOperation("GptNeox20BLayerEncoderOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputLayerNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &qkvLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionNode = opGraph_.nodes.at(nodeId++);
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
    
    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.dk = param_.dk;
    selfAttentionParam.headNum = param_.headNum;
    selfAttentionParam.model = "gptneox20b";
    selfAttentionParam.scalingFactor = param_.scalingFactor;
    
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
    positionEmbeddingNode.outTensorIds = {INTERMEDIATE_QUERYEMBED, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    selfAttentionNode.operation.reset(new AclTransformer::SelfAttentionOperation(selfAttentionParam));
    selfAttentionNode.inTensorIds = {INTERMEDIATE_QUERYEMBED, OUT_PRESENTKEY, OUT_PRESENTVALUE, IN_ATTENTIONMASK};
    selfAttentionNode.outTensorIds = {INTERMEDIATE_SELFATTNOUT};

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
    ffnResidualAddNode.outTensorIds = {OUT_GPTNEOXLARYEROUT};
}

GptNeox20BLayerEncoderOperation::~GptNeox20BLayerEncoderOperation() {}

uint64_t GptNeox20BLayerEncoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t GptNeox20BLayerEncoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status GptNeox20BLayerEncoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;  // [bs, sq, hn * hs]

    outTensorDescs.at(1) = inTensors.at(0).desc;  // [bs, hn, sq, hs]
    outTensorDescs.at(1).dims.clear();
    outTensorDescs.at(1).dims.push_back(inTensors.at(0).desc.dims.at(0));
    outTensorDescs.at(1).dims.push_back(param_.headNum);
    outTensorDescs.at(1).dims.push_back(inTensors.at(0).desc.dims.at(1));
    outTensorDescs.at(1).dims.push_back(param_.dk);

    outTensorDescs.at(2) = outTensorDescs.at(1);  // [bs, hn, sq, hs]
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer