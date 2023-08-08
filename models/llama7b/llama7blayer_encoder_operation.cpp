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
#include "llama7blayer_encoder_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/transpose_operation.h"

namespace AclTransformer {
enum LLaMA7BLayerEncoderTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QMIXDWEIGHT,
    IN_QMIXDBIAS,
    IN_KMIXDWEIGHT,
    IN_KMIXDBIAS,
    IN_VMIXDWEIGHT,
    IN_VMIXDBIAS,
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
    OUT_LLAMA7BLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 18;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 12;

LLaMA7BLayerEncoderOperation::LLaMA7BLayerEncoderOperation(const LLaMA7BLayerParam &param)
    : GraphOperation("LLaMA7BLayerEncoderOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdQLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdKLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mixdVLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &qPositionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &kPositionEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdQLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    mixdQLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QMIXDWEIGHT, IN_QMIXDBIAS};
    mixdQLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ};

    mixdKLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    mixdKLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_KMIXDWEIGHT, IN_KMIXDBIAS};
    mixdKLinearNode.outTensorIds = {INTERMIDATE_MIXEDK};

    mixdVLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    mixdVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_VMIXDWEIGHT, IN_VMIXDBIAS};
    mixdVLinearNode.outTensorIds = {INTERMIDATE_MIXEDV};

    qPositionEmbeddingNode.operation.reset(new AclTransformer::PositionEmbedding1dSplitOperation({param_.headNum}));
    qPositionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDQ, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
    qPositionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ};

    kPositionEmbeddingNode.operation.reset(new AclTransformer::PositionEmbedding1dSplitOperation({param_.headNum}));
    kPositionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDK, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
    kPositionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDK};

    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.dk = param_.dk;
    selfAttentionParam.headNum = param_.headNum;
    selfAttentionParam.model = param_.model;
    selfAttentionNode.operation.reset(
        new AclTransformer::SelfAttentionOperation(selfAttentionParam));
    selfAttentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_MIXEDV,
                                            IN_ATTENTIONMASK};
    selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    selfAttentionNode.inTensorViewFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorViewFuncs.at(2) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(1), oldDims.at(0), param_.headNum, oldDims.at(2) / param_.headNum};
    };

    selfOutLinearNode.operation.reset(new AclTransformer::LinearOperation({}));
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    selfResidualAddNode.inTensorViewFuncs.resize(selfResidualAddNode.inTensorIds.size());
    selfResidualAddNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(1), oldDims.at(0), oldDims.at(2)};
    };

    selfNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    mlpNode.operation.reset(new AclTransformer::MlpOperation({}));
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT, IN_MLPUPWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA7BLAYEROUT};
}

LLaMA7BLayerEncoderOperation::~LLaMA7BLayerEncoderOperation() {}

uint64_t LLaMA7BLayerEncoderOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t LLaMA7BLayerEncoderOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status LLaMA7BLayerEncoderOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                     AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = inTensors.at(0).desc;
    outTensorDescs.at(1).dims.at(0) = inTensors.at(0).desc.dims.at(1);
    outTensorDescs.at(1).dims.at(1) = inTensors.at(0).desc.dims.at(0);
    outTensorDescs.at(1).dims.at(2) = param_.headNum;
    outTensorDescs.at(1).dims.push_back(param_.dk);
    outTensorDescs.at(2) = outTensorDescs.at(1);
    
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer