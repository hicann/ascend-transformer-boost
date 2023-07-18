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
#include "bertlayer_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/add_norm_operation.h"

namespace AclTransformer {
enum BertLayerTensorId {
    IN_HIDDENSTATES_ID = 0,
    IN_QLINEARWEIGHT_ID,
    IN_LINEARBIAS_ID,
    IN_KLINEARWEIGHT_ID,
    IN_KLINEARBIAS_ID,
    IN_VLINEARWEIGHT_ID,
    IN_VLINEARBIAS_ID,
    IN_SELFOUTLINEARWEIGHT_ID,
    IN_SELFOUTLINEARBIAS_ID,
    IN_SELFOUTNORMWEIGHT_ID,
    IN_SELFOUTNORMBIAS_ID,
    IN_FFNLINEARWEIGHT_ID,
    IN_FFNLINEARBIAS_ID,
    IN_BERTOUTLINEARWEIGHT_ID,
    IN_BERTOUTLINEARBIAS_ID,
    IN_BERTOUTNORMWEIGHT_ID,
    IN_BERTOUTNORMBIAS_ID,
    IN_ATTENTIONMASK_ID,
    OUT_BERTLAYEROUT_ID,
    INTERMEDIATE_MIXEDQUERY_ID,
    INTERMEDIATE_MIXEDKEY_ID,
    INTERMEDIATE_MIXEDVALUE_ID,
    INTERMEDIATE_SELFATTENTIONOUT_ID,
    INTERMEDIATE_SELFLINEAROUT_ID,
    INTERMEDIATE_SELFADDNORMOUT_ID,
    INTERMEDIATE_FFNOUT_ID,
    INTERMEDIATE_BERTOUTLINEAROUT_ID,
};

static const uint64_t IN_TENSOR_COUNT = 18;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 9;

BertLayerOperation::BertLayerOperation(const BertLayerParam &param)
    : GraphOperation("BertLayerOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &qLinearNode = opGraph_.nodes.at(nodeId++);
    auto &kLinearNode = opGraph_.nodes.at(nodeId++);
    auto &vLinearNode = opGraph_.nodes.at(nodeId++);
    auto &selfAttentionNode = opGraph_.nodes.at(nodeId++);
    auto &selfOutLinearNode = opGraph_.nodes.at(nodeId++);
    auto &selfOutAddNormNode = opGraph_.nodes.at(nodeId++);
    auto &ffnNode = opGraph_.nodes.at(nodeId++);
    auto &bertOutLinearNode = opGraph_.nodes.at(nodeId++);
    auto &bertOutAddNormNode = opGraph_.nodes.at(nodeId++);

    qLinearNode.operation = std::make_shared<LinearOperation>(LinearParam());
    qLinearNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_QLINEARWEIGHT_ID, IN_LINEARBIAS_ID};
    qLinearNode.outTensorIds = {INTERMEDIATE_MIXEDQUERY_ID};

    kLinearNode.operation = std::make_shared<LinearOperation>(LinearParam());
    kLinearNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_KLINEARWEIGHT_ID, IN_KLINEARBIAS_ID};
    kLinearNode.outTensorIds = {INTERMEDIATE_MIXEDKEY_ID};

    vLinearNode.operation = std::make_shared<LinearOperation>(LinearParam());
    vLinearNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_VLINEARWEIGHT_ID, IN_VLINEARBIAS_ID};
    vLinearNode.outTensorIds = {INTERMEDIATE_MIXEDVALUE_ID};

    SelfAttentionParam selfAttentionParam;
    selfAttentionParam.transKey = param_.transKey;
    selfAttentionParam.dk = param_.dk;
    selfAttentionParam.headNum = param_.headNum;
    selfAttentionNode.operation = std::make_shared<SelfAttentionOperation>(selfAttentionParam);
    selfAttentionNode.inTensorIds = {INTERMEDIATE_MIXEDQUERY_ID, INTERMEDIATE_MIXEDKEY_ID, INTERMEDIATE_MIXEDVALUE_ID,
                                     IN_ATTENTIONMASK_ID};
    selfAttentionNode.outTensorIds = {INTERMEDIATE_SELFATTENTIONOUT_ID};

    selfOutLinearNode.operation = std::make_shared<LinearOperation>(LinearParam());
    selfOutLinearNode.inTensorIds = {INTERMEDIATE_SELFATTENTIONOUT_ID, IN_SELFOUTLINEARWEIGHT_ID,
                                     IN_SELFOUTLINEARBIAS_ID};
    selfOutLinearNode.outTensorIds = {INTERMEDIATE_SELFLINEAROUT_ID};

    selfOutAddNormNode.operation = std::make_shared<AddNormOperation>(AddNormParam());
    selfOutAddNormNode.inTensorIds = {INTERMEDIATE_SELFLINEAROUT_ID, IN_HIDDENSTATES_ID, IN_SELFOUTNORMWEIGHT_ID,
                                      IN_SELFOUTNORMBIAS_ID};
    selfOutAddNormNode.outTensorIds = {INTERMEDIATE_SELFADDNORMOUT_ID};

    ffnNode.operation = std::make_shared<FfnOperation>(FfnParam());
    ffnNode.inTensorIds = {INTERMEDIATE_SELFADDNORMOUT_ID, IN_FFNLINEARWEIGHT_ID, IN_FFNLINEARBIAS_ID};
    ffnNode.outTensorIds = {INTERMEDIATE_FFNOUT_ID};

    bertOutLinearNode.operation = std::make_shared<LinearOperation>(LinearParam());
    bertOutLinearNode.inTensorIds = {INTERMEDIATE_FFNOUT_ID, IN_BERTOUTLINEARWEIGHT_ID, IN_BERTOUTLINEARBIAS_ID};
    bertOutLinearNode.outTensorIds = {INTERMEDIATE_BERTOUTLINEAROUT_ID};

    bertOutAddNormNode.operation = std::make_shared<AddNormOperation>(AddNormParam());
    bertOutAddNormNode.inTensorIds = {INTERMEDIATE_BERTOUTLINEAROUT_ID, INTERMEDIATE_SELFADDNORMOUT_ID,
                                      IN_BERTOUTNORMWEIGHT_ID, IN_BERTOUTNORMBIAS_ID};
    bertOutAddNormNode.outTensorIds = {OUT_BERTLAYEROUT_ID};
}

BertLayerOperation::~BertLayerOperation() {}

uint64_t BertLayerOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t BertLayerOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status BertLayerOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer