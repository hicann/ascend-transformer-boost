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
#include "acltransformer/ops/lm_head_parallel_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/all_gather_operation.h"

namespace AclTransformer {
enum LmHeadParallelTensorId { IN_INPUT = 0, IN_WEIGHT, OUT_OUTPUT, INTERMEDIATE_LINEAR, INTERMEDIATE_ALLGATHER };

static uint64_t IN_TENSOR_COUNT = 2;
static uint64_t OUT_TENSOR_COUNT = 1;
static uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static uint64_t NODE_COUNT = 3;

LmHeadParallelOperation::LmHeadParallelOperation(const LmHeadParallelParam &param) : GraphOperation("LmHeadParallelOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &linearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &allGatherNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &transposeNode = opGraph_.nodes.at(nodeId++);

    linearNode.operation.reset(new AclTransformer::LinearOperation({false, false, false}));
    linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
    linearNode.outTensorIds = {INTERMEDIATE_LINEAR};
    linearNode.inTensorViewFuncs.resize(linearNode.inTensorIds.size());

    allGatherNode.operation.reset(new AclTransformer::AllGatherOperation({param_.rank, param_.rankSize, param_.rankRoot, param_.backend}));
    allGatherNode.inTensorIds = {INTERMEDIATE_LINEAR};
    allGatherNode.outTensorIds = {INTERMEDIATE_ALLGATHER};

    transposeNode.operation.reset(new AclTransformer::TransposeOperation({param_.perm}));
    transposeNode.inTensorIds = {INTERMEDIATE_ALLGATHER};
    transposeNode.outTensorIds = {OUT_OUTPUT};
}

LmHeadParallelOperation::~LmHeadParallelOperation() {}

uint64_t LmHeadParallelOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t LmHeadParallelOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status LmHeadParallelOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                               AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[1], inTensors.at(0).desc.dims[0],
                                 inTensors.at(1).desc.dims[0] * param_.rankSize};

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer