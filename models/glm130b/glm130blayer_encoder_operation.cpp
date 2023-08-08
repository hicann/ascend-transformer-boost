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
#include "glm130blayer_encoder_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/linear_parallel_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/mlp_operation.h"

namespace AclTransformer {
enum glm130BLayerEncoderTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTNORMBIAS,
    IN_MLPLINEARWEIGHT,
    IN_MLPLINEARBIAS,
    IN_MLPOUTLINEARWEIGHT,
    IN_MLPOUTLINEARBIAS,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    OUT_GLMLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
    INTERMIDATE_MLPLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 17;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 9;
static const uint64_t NODE_COUNT = 10;

Glm130BLayerEncoderOperation::Glm130BLayerEncoderOperation(const Glm130BLayerParam &param)
    : GraphOperation("Glm130BLayerEncoderOperation"), param_(param)
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
    GraphOperation::Node &selfOutLinearParallelNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpLinearParallelNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdQkvLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS};
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    positionEmbeddingNode.operation.reset(
        new AclTransformer::PositionEmbeddingOperation({false, param_.headNum / param_.rankSize}));
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.transKey = param_.transKey;
    selfAttentionParam.dk = param_.dk;
    selfAttentionParam.headNum = param_.headNum / param_.rankSize;
    selfAttentionParam.layerId = param_.layerId;
    selfAttentionParam.model = "chatglm6b";
    selfAttentionNode.operation.reset(new AclTransformer::SelfAttentionOperation(selfAttentionParam));
    selfAttentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, OUT_PRESENTKEY, OUT_PRESENTVALUE, IN_ATTENTIONMASK};
    selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT};

    selfOutLinearParallelNode.operation.reset(new AclTransformer::LinearParallelOperation(
        {false, param_.rank, param_.rankSize, 0, "", "RowParallel", param_.backend}));
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    selfResidualAddNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    selfNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    mlpNode.operation.reset(new AclTransformer::MlpOperation({"glm130b"}));
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPLINEARWEIGHT, IN_MLPLINEARBIAS};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    mlpLinearParallelNode.operation.reset(new AclTransformer::LinearParallelOperation(
        {false, param_.rank, param_.rankSize, 0, "", "RowParallel", param_.backend}));
    mlpLinearParallelNode.inTensorIds = {INTERMIDATE_MLPOUT, IN_MLPOUTLINEARWEIGHT, IN_MLPOUTLINEARBIAS};
    mlpLinearParallelNode.outTensorIds = {INTERMIDATE_MLPLINEAROUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({param_.residualAddScale}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, INTERMIDATE_MLPLINEAROUT};
    mlpResidualAddNode.outTensorIds = {OUT_GLMLAYEROUT};
}

Glm130BLayerEncoderOperation::~Glm130BLayerEncoderOperation() {}

uint64_t Glm130BLayerEncoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t Glm130BLayerEncoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status Glm130BLayerEncoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    size_t tensorId = 0;
    outTensorDescs.at(tensorId) = inTensors.at(0).desc;
    tensorId++;
    outTensorDescs.at(tensorId) = inTensors.at(0).desc;
    outTensorDescs.at(tensorId).dims.at(2) = param_.headNum / param_.rankSize;
    outTensorDescs.at(tensorId).dims.push_back(param_.dk);
    tensorId++;
    outTensorDescs.at(tensorId) = outTensorDescs.at(1);
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer