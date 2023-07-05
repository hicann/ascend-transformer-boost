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
#include "chatglm6b_operation.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
enum Chatglm6BTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT = 1,
    IN_NORMBIAS = 2,
    IN_QKVMIXDWEIGHT = 3,
    IN_QKVMIXDBIAS = 4,
    IN_SELFOUTLINEARWEIGHT = 5,
    IN_SELFOUTLINEARBIAS = 6,
    IN_SELFOUTNORMWEIGHT = 7,
    IN_SELFOUTNORMBIAS = 8,
    IN_FFNLINEARWEIGHT = 9,
    IN_FFNLINEARBIAS = 10,
    IN_FFNOUTLINEARWEIGHT = 11,
    IN_FFNOUTLINEARBIAS = 12,
    IN_POSITIONIDS = 13,
    IN_COSTABLE = 14,
    IN_SINTABLE = 15,
    IN_ATTENTIONMASK = 16,
    IN_PASTKEY = 17,
    IN_PASTVALUE = 18,
    OUT_GLMBLOCKOUT = 19,
    OUT_PRESENTKEY = 20,
    OUT_PRESENTVALUE = 21,
    INTERMIDATE_INPUTNORMOUT = 22,
    INTERMIDATE_MIXEDLINEAROUTQKV = 23,
    INTERMIDATE_POSITIONEMBEDQ = 24,
    INTERMIDATE_POSITIONEMBEDK = 25,
    INTERMIDATE_VALUE = 26,
    INTERMIDATE_SELFOUT = 27,
    INTERMIDATE_SELFLINEAROUT = 28,
    INTERMIDATE_SELFRESIDUALADDOUT = 29,
    INTERMIDATE_SELFNORMOUT = 30,
    INTERMIDATE_FFNOUT = 31,
    INTERMIDATE_FFNLINEAROUT = 32,
};

static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

ChatGlm6BOperation::ChatGlm6BOperation(ChatGlm6BParam &param)
    : GraphOperation("ChatGlm6BOperation"), param_(param)
{
    BuildGraph();
}

ChatGlm6BOperation::~ChatGlm6BOperation() {}

uint64_t ChatGlm6BOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm6BOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status ChatGlm6BOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    const AsdOps::Tensor &keyTensor = inTensors.at(IN_PASTKEY);
    const AsdOps::Tensor &valueTensor = inTensors.at(IN_PASTVALUE);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = keyTensor.desc;
    outTensorDescs.at(1).dims.at(0) += 1;
    outTensorDescs.at(2) = valueTensor.desc;
    outTensorDescs.at(2).dims.at(0) += 1;
    return AsdOps::Status::OkStatus();
}

void ChatGlm6BOperation::BuildGraph()
{
    operationGraph_.inTensorSize = IN_TENSOR_COUNT;
    operationGraph_.outTensorSize = OUT_TENSOR_COUNT;
    const int intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    operationGraph_.intermediateTensorSize = intermediateTensorSize;
    operationGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    AclTransformer::OperationGraphNode &inputNormNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &mixdQkvLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfOutLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfResidualAddNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfNormNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &ffnNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &ffnLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &ffnResidualAddNode = operationGraph_.nodes.at(nodeId++);

    inputNormNode.operation = new AclTransformer::NormOperation({param_.layerNormEps});
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    mixdQkvLinearNode.operation = new AclTransformer::LinearOperation({});
    mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS};
    mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

    positionEmbeddingNode.operation = new AclTransformer::PositionEmbeddingOperation({param_.headNum});
    positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE};
    positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_VALUE};

    selfAttentionKvCacheNode.operation = new AclTransformer::SelfAttentionKvCacheOperation(
        {param_.transKey, param_.transKey, param_.headNum, param_.layerId});
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_VALUE,
                                            IN_ATTENTIONMASK,
                                            IN_PASTKEY,
                                            IN_PASTVALUE};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT, OUT_PRESENTKEY, OUT_PRESENTVALUE};

    selfOutLinearNode.operation = new AclTransformer::LinearOperation({});
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    selfResidualAddNode.operation = new AclTransformer::AddOperation({param_.residualAddScale});
    selfResidualAddNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    selfNormNode.operation = new AclTransformer::NormOperation({param_.layerNormEps});
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    ffnNode.operation = new AclTransformer::FfnOperation({});
    ffnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS};
    ffnNode.outTensorIds = {INTERMIDATE_FFNOUT};

    ffnLinearNode.operation = new AclTransformer::LinearOperation({});
    ffnLinearNode.inTensorIds = {INTERMIDATE_FFNOUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS};
    ffnLinearNode.outTensorIds = {INTERMIDATE_FFNLINEAROUT};

    ffnResidualAddNode.operation = new AclTransformer::AddOperation({param_.residualAddScale});
    ffnResidualAddNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, INTERMIDATE_FFNLINEAROUT};
    ffnResidualAddNode.outTensorIds = {OUT_GLMBLOCKOUT};
}
} // namespace AclTransformer
