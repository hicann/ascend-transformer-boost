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
#include "chatglm6b_lmhead_operation.h"
#include <atb/atb_infer.h>

namespace atb_speed {

enum Chatglm6BLayerLmHeadTensorId {
    IN_HIDDENSTATES = 0,
    IN_LINEARWEIGHT,
    OUT_LMHEAD_OUT,
    INTERMIDATE_LINEAR_OUT
};

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;
static uint64_t DIM3 = 3;

atb::Status CreateChatGlm6BLmHeadOperation(const ChatGlm6BLmHeadParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &transposeNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearParam = {false, false, false};
    CreateOp(linearParam, &linearNode.op);
    linearNode.inTensorIds = {IN_HIDDENSTATES, IN_LINEARWEIGHT};
    linearNode.outTensorIds = {INTERMIDATE_LINEAR_OUT};

    atb::infer::TransposeParam transposeParam;
    transposeParam.perm = {1, 0, 2};
    CreateOp(transposeParam, &transposeNode.op);
    transposeNode.inTensorIds = {INTERMIDATE_LINEAR_OUT};
    transposeNode.outTensorIds = {OUT_LMHEAD_OUT};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape.dimNum = DIM3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[0]; // index: 2
        ATB_LOG(INFO) << "LmHead infershape success";
        return atb::NO_ERROR;
    };

    atb::CreateOp(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace atb_speed