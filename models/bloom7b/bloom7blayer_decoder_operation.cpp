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
#include "bloom7blayer_decoder_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
enum Bloom7BLayerDecoderTensorId {
    IN_NORM_WEIGHT = 0,
    IN_NORM_BIAS,
    IN_QKVMIXD_WEIGHT,
    IN_QKVMIXD_BIAS,
    IN_DENSE_WEIGHT,
    IN_DENSE_BIAS,
    IN_SELFOUTNORM_WEIGHT,
    IN_SELFOUTNORM_BIAS,
    IN_HTO4H_WEIGHT,
    IN_HTO4H_BIAS,
    IN_4HTOH_WEIGHT,
    IN_4HTOH_BIAS,
    IN_HIDDEN_STATES,
    IN_ALIBI,
    IN_ATTENTION_MASK,
    IN_PAST_KEY,
    IN_PAST_VALUE,
    OUT_LAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORM_OUT,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_FFNOUT,
    INTERMIDATE_MLPLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 17;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 6;
static const uint64_t HIDDEN_STATES_DIM = 3;

Bloom7BLayerDecoderOperation::Bloom7BLayerDecoderOperation(const Bloom7BLayerParam &param)
    : GraphOperation("Bloom7BLayerDecoderOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionKvCacheNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpFfnNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT, IN_NORM_BIAS};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORM_OUT};

    SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param_.headNum;
    selfAttentionKvCacheParam.dk = param_.dk;
    selfAttentionKvCacheParam.invNormFactorvarAttr = param_.invNormFactorvarAttr;
    selfAttentionKvCacheParam.model = param_.model;
    selfAttentionKvCacheNode.operation.reset(
        new AclTransformer::SelfAttentionKvCacheOperation(selfAttentionKvCacheParam));
    selfAttentionKvCacheNode.inTensorIds = {
        INTERMIDATE_INPUTNORM_OUT, IN_QKVMIXD_WEIGHT, IN_QKVMIXD_BIAS,   IN_PAST_KEY,     IN_PAST_VALUE, IN_ALIBI,
        IN_DENSE_WEIGHT,           IN_DENSE_BIAS,     IN_ATTENTION_MASK, IN_HIDDEN_STATES};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    selfNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    FfnParam mlpFfnParam;
    mlpFfnParam.activationFuncType = FfnParam::ActivationFuncType(param_.activationFuncType);
    mlpFfnNode.operation.reset(new AclTransformer::FfnOperation(mlpFfnParam));
    mlpFfnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_HTO4H_WEIGHT, IN_HTO4H_BIAS};
    mlpFfnNode.outTensorIds = {INTERMIDATE_FFNOUT};

    mlpLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    mlpLinearNode.inTensorIds = {INTERMIDATE_FFNOUT, IN_4HTOH_WEIGHT, IN_4HTOH_BIAS};
    mlpLinearNode.outTensorIds = {INTERMIDATE_MLPLINEAROUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_MLPLINEAROUT, INTERMIDATE_SELFOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};
}

Bloom7BLayerDecoderOperation::~Bloom7BLayerDecoderOperation() {}

uint64_t Bloom7BLayerDecoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t Bloom7BLayerDecoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status Bloom7BLayerDecoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    const AsdOps::Tensor &keyTensor = inTensors.at(IN_PAST_KEY);
    const AsdOps::Tensor &valueTensor = inTensors.at(IN_PAST_VALUE);
    outTensorDescs.at(0) = inTensors.at(IN_HIDDEN_STATES).desc;
    outTensorDescs.at(1) = keyTensor.desc;
    outTensorDescs.at(1).dims.at(2) += 1;
    const size_t tensorId2 = 2;
    outTensorDescs.at(tensorId2) = valueTensor.desc;
    outTensorDescs.at(tensorId2).dims.at(1) += 1;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer