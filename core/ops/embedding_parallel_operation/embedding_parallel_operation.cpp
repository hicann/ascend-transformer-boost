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
#include "acltransformer/ops/embedding_parallel_operation.h"
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/all_gather_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include <asdops/utils/log/log.h>
namespace AclTransformer {
enum EmbeddingParallelTensorId {
    IN_WORDTABLE = 0,
    IN_INPUTIDS,
    OUT_WORDEMBEDDINGOUT,
    INTERMIDATE_EMBEDDEDOUT,
    INTERMIDATE_GATHEREDOUT
};

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;

EmbeddingParallelOperation::EmbeddingParallelOperation(const EmbeddingParallelParam &param)
    : GraphOperation("EmbeddingParallelOperation"), param_(param)
{
    ASD_LOG(INFO) << " EmbeddingParallelOperation::EmbeddingParallelOperation called";
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &embeddingNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &allGatherNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &transposeNode = opGraph_.nodes.at(nodeId++);

    embeddingNode.operation.reset(new AclTransformer::EmbeddingOperation({param_.axis}));
    embeddingNode.inTensorIds = {IN_WORDTABLE, IN_INPUTIDS};
    embeddingNode.outTensorIds = {INTERMIDATE_EMBEDDEDOUT};

    allGatherNode.operation.reset(new AclTransformer::AllGatherOperation({param_.rank, param_.rankSize, param_.rankRoot, param_.backend}));
    allGatherNode.inTensorIds = {INTERMIDATE_EMBEDDEDOUT};
    allGatherNode.outTensorIds = {INTERMIDATE_GATHEREDOUT};

    transposeNode.operation.reset(new AclTransformer::TransposeOperation({param_.perm}));
    transposeNode.inTensorIds = {INTERMIDATE_GATHEREDOUT};
    transposeNode.outTensorIds = {OUT_WORDEMBEDDINGOUT};
}

EmbeddingParallelOperation::~EmbeddingParallelOperation() {}

uint64_t EmbeddingParallelOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t EmbeddingParallelOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status EmbeddingParallelOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                     AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    const AsdOps::Tensor &wordTableTensor = inTensors.at(0);
    const AsdOps::Tensor &inputIdsTensor = inTensors.at(1);
    outTensorDescs.at(0) = wordTableTensor.desc;
    outTensorDescs.at(0).dims = {inputIdsTensor.desc.dims[1], inputIdsTensor.desc.dims[0], wordTableTensor.desc.dims[1] * param_.rankSize};
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer