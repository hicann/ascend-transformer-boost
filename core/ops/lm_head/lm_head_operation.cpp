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
#include "acltransformer/ops/lm_head_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/transpose_operation.h"

namespace AclTransformer {
enum LmHeadTensorId { IN_INPUT = 0, IN_WEIGHT, OUT_OUTPUT, INTERMEDIATE_LINEAR };

static uint64_t IN_TENSOR_COUNT = 2;
static uint64_t OUT_TENSOR_COUNT = 1;
static uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static uint64_t NODE_COUNT = 2;

LmHeadOperation::LmHeadOperation(const LmHeadParam &param) : GraphOperation("LmHeadOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    GraphOperation::Node &linearNode = opGraph_.nodes.at(nodeId++);
    GraphOperation::Node &transposeNode = opGraph_.nodes.at(nodeId++);

    linearNode.operation.reset(new AclTransformer::LinearOperation({false, false, false}));
    linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
    linearNode.outTensorIds = {INTERMEDIATE_LINEAR};
    linearNode.inTensorViewFuncs.resize(linearNode.inTensorIds.size());

    transposeNode.operation.reset(new AclTransformer::TransposeOperation({{1, 0, 2}}));
    transposeNode.inTensorIds = {INTERMEDIATE_LINEAR};
    transposeNode.outTensorIds = {OUT_OUTPUT};
}

LmHeadOperation::~LmHeadOperation() {}

uint64_t LmHeadOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t LmHeadOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status LmHeadOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                               AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[1], inTensors.at(0).desc.dims[0],
                                 inTensors.at(1).desc.dims[0]};

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer