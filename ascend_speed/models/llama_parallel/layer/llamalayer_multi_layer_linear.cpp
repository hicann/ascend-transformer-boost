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
#include "llamalayer_multi_layer_linear.h"
#include <atb/atb_infer.h>

namespace atb_speed {

enum LlamamultiLayerLinearTensorId {
    IN_INPUTTENSOR = 0,
    IN_WEIGHTTENSOR,
    OUT_MATMULRESULTQTENSOR,
    OUT_MATMULRESULTKTENSOR,
    OUT_MATMULRESULTVTENSOR,
    INTERMIDATE_LINEAR_OUT
};

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;
static uint64_t DIM3 = 3;

atb::Status CreateLlamaMultiLayerLinearOperation(const MultiLayerLinearParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearParam = {false, false, false};
    CreateOp(linearParam, &linearNode.op);
    linearNode.inTensorIds = {IN_INPUTTENSOR, IN_WEIGHTTENSOR};
    linearNode.outTensorIds = {INTERMIDATE_LINEAR_OUT};

    atb::infer::SplitParam splitParam = {2, 3};
    CreateOp(splitParam, &splitNode.op);
    splitNode.inTensorIds = {INTERMIDATE_LINEAR_OUT};
    splitNode.outTensorIds = {OUT_MATMULRESULTQTENSOR, OUT_MATMULRESULTKTENSOR, OUT_MATMULRESULTVTENSOR};
    splitNode.inTensorReshapeFuncs.resize(splitNode.inTensorIds.size());
    splitNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = DIM3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / DIM3;

        outTensorDescs.at(1) = inTensorDescs.at(0);
        outTensorDescs.at(1).shape.dimNum = DIM3;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / DIM3;

        outTensorDescs.at(2) = inTensorDescs.at(0);
        outTensorDescs.at(2).shape.dimNum = DIM3;
        outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(2).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(2).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / DIM3;

        return atb::NO_ERROR;
    };

    atb::CreateOp(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace atb_speed