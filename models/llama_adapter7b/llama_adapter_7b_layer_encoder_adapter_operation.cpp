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
#include "llama_adapter_7b_layer_encoder_adapter_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/apply_rotary_emb_operation.h"
#include "acltransformer/ops/self_attention_cross_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/transpose_operation.h"

namespace AclTransformer {
enum LLaMAAdapter7BLayerEncoderAdapterTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QWEIGHT,
    IN_QBIAS,
    IN_KWEIGHT,
    IN_VWEIGHT,
    IN_GATETANH,
    IN_OWEIGHT,
    IN_OBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPW1WEIGHT,
    IN_MLPW1BIAS,
    IN_MLPW2WEIGHT,
    IN_MLPW2BIAS,
    IN_MLPW3WEIGHT,
    IN_MLPW3BIAS,
    IN_FREQSCIS,
    IN_ADAPTER,
    IN_ATTENTIONMASK,
    OUT_LLAMA7BLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_Q,
    INTERMIDATE_K,
    INTERMIDATE_V,
    INTERMIDATE_ROPEQ,
    INTERMIDATE_ROPEK,
    INTERMIDATE_AV,
    INTERMIDATE_AK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 13;
static const uint64_t NODE_COUNT = 13;

LLaMAAdapter7BLayerEncoderAdapterOperation::LLaMAAdapter7BLayerEncoderAdapterOperation(const LLaMAAdapter7BLayerParam &param)
    : GraphOperation("LLaMAAdapter7BLayerEncoderAdapterOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &wQLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &wKLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &wVLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &ropeNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &wAVLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &wAKLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfAttentionDeNoAdapterNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &wOLinearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfResidualAddNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &selfNormNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &mlpResidualAddNode = opGraph_.nodes.at(nodeId++);

    inputNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    wQLinearNode.operation.reset(new AclTransformer::LinearOperation({false, false, true}));
    wQLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QWEIGHT, IN_QBIAS};
    wQLinearNode.outTensorIds = {INTERMIDATE_Q};

    wKLinearNode.operation.reset(new AclTransformer::LinearOperation({false, false, false}));
    wKLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_KWEIGHT};
    wKLinearNode.outTensorIds = {INTERMIDATE_K};

    wVLinearNode.operation.reset(new AclTransformer::LinearOperation({false, false, false}));
    wVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_VWEIGHT};
    wVLinearNode.outTensorIds = {INTERMIDATE_V};

    ropeNode.operation.reset(new AclTransformer::ApplyRotaryEmbOperation({}));
    ropeNode.inTensorIds = {INTERMIDATE_Q, INTERMIDATE_K, IN_FREQSCIS};
    ropeNode.outTensorIds = {INTERMIDATE_ROPEQ, INTERMIDATE_ROPEK};
    ropeNode.inTensorViewFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };
    ropeNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };

    wAVLinearNode.operation.reset(new AclTransformer::LinearOperation({false, false, false}));
    wAVLinearNode.inTensorIds = {IN_ADAPTER, IN_VWEIGHT};
    wAVLinearNode.outTensorIds = {INTERMIDATE_AV};

    wAKLinearNode.operation.reset(new AclTransformer::LinearOperation({false, false, false}));
    wAKLinearNode.inTensorIds = {IN_ADAPTER, IN_KWEIGHT};
    wAKLinearNode.outTensorIds = {INTERMIDATE_AK};

    AclTransformer::SelfAttentionCrossParam selfAttentionCrossParam;
    selfAttentionCrossParam.dk = param_.dk;
    selfAttentionCrossParam.headNum = param_.headNum;
    selfAttentionCrossParam.model = param_.model;
    selfAttentionDeNoAdapterNode.operation.reset(
        new AclTransformer::SelfAttentionCrossOperation(selfAttentionCrossParam));
    selfAttentionDeNoAdapterNode.inTensorIds = {INTERMIDATE_ROPEQ,
                                                INTERMIDATE_ROPEK,
                                                INTERMIDATE_V,
                                                INTERMIDATE_AV,
                                                INTERMIDATE_AK,
                                                IN_GATETANH,
                                                IN_ATTENTIONMASK};
    selfAttentionDeNoAdapterNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    selfAttentionDeNoAdapterNode.inTensorViewFuncs.resize(selfAttentionDeNoAdapterNode.inTensorIds.size());
    selfAttentionDeNoAdapterNode.inTensorViewFuncs.at(2) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };
    selfAttentionDeNoAdapterNode.inTensorViewFuncs.at(3) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };
    selfAttentionDeNoAdapterNode.inTensorViewFuncs.at(4) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };

    wOLinearNode.operation.reset(new AclTransformer::LinearOperation({false, false, true}));
    wOLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_OWEIGHT, IN_OBIAS};
    wOLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    
    selfNormNode.operation.reset(new AclTransformer::RmsNormOperation({param_.rmsNormEps}));
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    AclTransformer::MlpParam mlpParam;
    mlpParam.model = "llama_adapter";
    mlpNode.operation.reset(new AclTransformer::MlpOperation({mlpParam}));
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPW1WEIGHT, IN_MLPW1BIAS, IN_MLPW2WEIGHT, IN_MLPW2BIAS, IN_MLPW3WEIGHT, IN_MLPW3BIAS};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    mlpResidualAddNode.operation.reset(new AclTransformer::AddOperation({}));
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA7BLAYEROUT};
}

LLaMAAdapter7BLayerEncoderAdapterOperation::~LLaMAAdapter7BLayerEncoderAdapterOperation() {}

uint64_t LLaMAAdapter7BLayerEncoderAdapterOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t LLaMAAdapter7BLayerEncoderAdapterOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status LLaMAAdapter7BLayerEncoderAdapterOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
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