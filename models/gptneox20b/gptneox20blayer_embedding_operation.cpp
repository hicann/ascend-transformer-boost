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
#include "gptneox20blayer_embedding_operation.h"
#include "acltransformer/ops/embedding_operation.h"

namespace AclTransformer {
enum GptNeox20BLayerEmbeddingTensorId {
    IN_EMBEDDING_WEIGHTS = 0,
    IN_INPUT_IDS,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_POSITION_IDS,
    OUT_HIDDEN_STATES,
    OUT_COS_EMBED,
    OUT_SIN_EMBED,
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 3;

GptNeox20BLayerEmbeddingOperation::GptNeox20BLayerEmbeddingOperation(const GptNeox20BLayerEmbeddingParam &param)
    : GraphOperation("GptNeox20BLayerEmbeddingOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &inputIdEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &cosEmbeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &sinEmbeddingNode = opGraph_.nodes.at(nodeId++);

    // params
    AclTransformer::EmbeddingParam embeddingParam;
    embeddingParam.axis = param_.axis;

    inputIdEmbeddingNode.operation.reset(new AclTransformer::EmbeddingOperation(embeddingParam));
    inputIdEmbeddingNode.inTensorIds = {IN_EMBEDDING_WEIGHTS, IN_INPUT_IDS};
    inputIdEmbeddingNode.outTensorIds = {OUT_HIDDEN_STATES};

    cosEmbeddingNode.operation.reset(new AclTransformer::EmbeddingOperation(embeddingParam));
    cosEmbeddingNode.inTensorIds = {IN_COS_TABLE, IN_POSITION_IDS};
    cosEmbeddingNode.outTensorIds = {OUT_COS_EMBED};

    sinEmbeddingNode.operation.reset(new AclTransformer::EmbeddingOperation(embeddingParam));
    sinEmbeddingNode.inTensorIds = {IN_SIN_TABLE, IN_POSITION_IDS};
    sinEmbeddingNode.outTensorIds = {OUT_SIN_EMBED};
}

GptNeox20BLayerEmbeddingOperation::~GptNeox20BLayerEmbeddingOperation() {}

uint64_t GptNeox20BLayerEmbeddingOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t GptNeox20BLayerEmbeddingOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status GptNeox20BLayerEmbeddingOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;  // [bs, sq, hidden_size]
    outTensorDescs.at(0).dims.clear();
    outTensorDescs.at(0).dims.push_back(inTensors.at(1).desc.dims[0]);
    outTensorDescs.at(0).dims.push_back(inTensors.at(1).desc.dims[1]);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[1]);

    outTensorDescs.at(1) = inTensors.at(2).desc;  // [bs, sq, rd]
    outTensorDescs.at(1).dims.clear();
    outTensorDescs.at(1).dims.push_back(inTensors.at(4).desc.dims[0]);
    outTensorDescs.at(1).dims.push_back(inTensors.at(4).desc.dims[1]);
    outTensorDescs.at(1).dims.push_back(inTensors.at(2).desc.dims[1]);

    outTensorDescs.at(2) = outTensorDescs.at(1);  // [bs, sq, rd]
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer