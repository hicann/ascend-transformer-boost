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
#include "baichuan1_7b_layer_encoder_with_bias_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/transpose_operation.h"

namespace AclTransformer {
enum BaiChuan17BLayerEncoderWithBiasTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXEDLINEARWEIGHT,
    IN_QKVMIXEDLINEARBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_MLPUPWEIGHT,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    OUT_BAICHUAN17BLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVMIXEDLINEAROUT,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 14;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 9;

BaiChuan17BLayerEncoderWithBiasOperation::BaiChuan17BLayerEncoderWithBiasOperation(const BaiChuan17BLayerParam &param)
    : GraphOperation("BaiChuan17BLayerEncoderWithBiasOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &qkvLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &positionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    qkvLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    qkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT, IN_QKVMIXEDLINEARBIAS};
    qkvLinearNode.outTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};

    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.model = param_.model;
    positionEmbeddingParam.headNum = param_.headNum;
    positionEmbeddingNode.operation.reset(new AclTransformer::PositionEmbeddingOperation(positionEmbeddingParam));
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.dk = param_.dk;
    selfAttentionParam.headNum = param_.headNum;
    selfAttentionParam.model = "baichuan1_7b";
    selfAttentionNode.operation.reset(new AclTransformer::SelfAttentionOperation(selfAttentionParam));
    selfAttentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, OUT_PRESENTKEY, OUT_PRESENTVALUE, IN_ATTENTIONMASK};
    selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT};

    selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    selfResidualAddNode.inTensorViewFuncs.resize(selfResidualAddNode.inTensorIds.size());

    selfNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    mlpNode.operation.reset(new AclTransformer::MlpOperation({}));
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT, IN_MLPUPWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_BAICHUAN17BLAYEROUT};
}

BaiChuan17BLayerEncoderWithBiasOperation::~BaiChuan17BLayerEncoderWithBiasOperation() {}

uint64_t BaiChuan17BLayerEncoderWithBiasOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t BaiChuan17BLayerEncoderWithBiasOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status
BaiChuan17BLayerEncoderWithBiasOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                 AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = inTensors.at(0).desc;
    outTensorDescs.at(1).dims.at(0) = inTensors.at(0).desc.dims.at(0);
    outTensorDescs.at(1).dims.at(1) = inTensors.at(0).desc.dims.at(1);
    outTensorDescs.at(1).dims.at(2) = param_.headNum;
    outTensorDescs.at(1).dims.push_back(param_.dk);
    outTensorDescs.at(2) = outTensorDescs.at(1);
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer