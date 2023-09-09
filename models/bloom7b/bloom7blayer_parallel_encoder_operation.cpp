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
#include "bloom7blayer_parallel_encoder_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/linear_parallel_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
enum Bloom7BLayerParallelDecoderTensorId {
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
    OUT_LAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORM_OUT,
    INTERMIDATE_QKVLINEAR_OUT,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_FFNOUT,
    INTERMIDATE_MLPLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 15;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 9;
static const uint64_t HIDDEN_STATES_DIM = 3;

Bloom7BLayerParallelEncoderOperation::Bloom7BLayerParallelEncoderOperation(const Bloom7BLayerParallelParam &param)
    : GraphOperation("Bloom7BLayerParallelEncoderOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &fusedQKVNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpFfnNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDEN_STATES, IN_NORM_WEIGHT, IN_NORM_BIAS};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORM_OUT};

    fusedQKVNode.operation.reset(new AclTransformer::LinearOperation({}));
    fusedQKVNode.inTensorIds = {INTERMIDATE_INPUTNORM_OUT, IN_QKVMIXD_WEIGHT, IN_QKVMIXD_BIAS};
    fusedQKVNode.outTensorIds = {INTERMIDATE_QKVLINEAR_OUT};

    SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param_.headNum;
    selfAttentionParam.dk = param_.dk;
    selfAttentionParam.invNormFactorvarAttr = param_.invNormFactorvarAttr;
    selfAttentionParam.model = param_.model;
    selfAttentionNode.operation.reset(
        new AclTransformer::SelfAttentionOperation(selfAttentionParam));
    selfAttentionNode.inTensorIds = {INTERMIDATE_QKVLINEAR_OUT, IN_ALIBI, IN_ATTENTION_MASK};
    selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    selfOutLinearNode.operation.reset(new AclTransformer::LinearParallelOperation(
        {false, param_.rank, param_.rankSize, 0, "yes", "RowParallel", "hccl", false, nullptr, true}));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_DENSE_WEIGHT, IN_DENSE_BIAS};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfOutAddNode.operation.reset(new AclTransformer::AddOperation({}));
    selfOutAddNode.inTensorIds = {INTERMIDATE_SELFLINEAROUT, IN_HIDDEN_STATES};
    selfOutAddNode.outTensorIds = {INTERMIDATE_SELFADDOUT};

    selfNormNode.operation.reset(new AclTransformer::NormOperation({param_.layerNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFADDOUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    FfnParam mlpFfnParam;
    mlpFfnParam.activationFuncType = FfnParam::ActivationFuncType(param_.activationFuncType);
    mlpFfnNode.operation.reset(new AclTransformer::FfnOperation(mlpFfnParam));
    mlpFfnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_HTO4H_WEIGHT, IN_HTO4H_BIAS};
    mlpFfnNode.outTensorIds = {INTERMIDATE_FFNOUT};

    mlpLinearNode.operation.reset(new AclTransformer::LinearParallelOperation(
        {false, param_.rank, param_.rankSize, 0, "yes", "RowParallel", "hccl", false, nullptr, true}));
    mlpLinearNode.inTensorIds = {INTERMIDATE_FFNOUT, IN_4HTOH_WEIGHT, IN_4HTOH_BIAS};
    mlpLinearNode.outTensorIds = {INTERMIDATE_MLPLINEAROUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_MLPLINEAROUT, INTERMIDATE_SELFADDOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};
}

Bloom7BLayerParallelEncoderOperation::~Bloom7BLayerParallelEncoderOperation() {}

uint64_t Bloom7BLayerParallelEncoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t Bloom7BLayerParallelEncoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status Bloom7BLayerParallelEncoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(IN_HIDDEN_STATES).desc;
    outTensorDescs.at(1) = inTensors.at(IN_HIDDEN_STATES).desc;
    outTensorDescs.at(1).dims = {
        inTensors.at(IN_HIDDEN_STATES).desc.dims[0] * param_.headNum, param_.dk, inTensors.at(IN_HIDDEN_STATES).desc.dims[1]};
    outTensorDescs.at(2) = inTensors.at(IN_HIDDEN_STATES).desc;
    outTensorDescs.at(2).dims = {
        inTensors.at(IN_HIDDEN_STATES).desc.dims[0] * param_.headNum,
        inTensors.at(IN_HIDDEN_STATES).desc.dims[1],
        param_.dk
    };
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer