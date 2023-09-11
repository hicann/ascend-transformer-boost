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
#include "chatglm6b_encoder_rope_model.h"
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "chatglm6b/layer/chatglm6blayer_encoder_operation.h"
#include "atb_speed/utils/tensor_util.h"

namespace atb_speed {
const int WEIGHT_COUNT_PER_LAYER = 12;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 2;
const int OPERATION_COUNT_BEFORE_LAYER = 2;
const int OPERATION_COUNT_AFTER_LAYER = 1;

enum InTensorId {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX,
};

void ChatGlm6BEncoderRopeModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    if (paramJson.contains("beginNormAxis")) {
        beginNormAxis = paramJson["beginNormAxis"].get<int>();
    }
    ATB_LOG(INFO) << "ChatGlm6BEncoderRopeModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", transKey:" << transKey << ", dk:" << dk << ", layerNum:" << layerNum
                  << ", residualAddScale:" << residualAddScale << ", beginNormAxis:" << beginNormAxis;
}

ChatGlm6BEncoderRopeModel::ChatGlm6BEncoderRopeModel(const std::string &param)
    : Model("ChatGlm6BEncoderRopeModel", param)
{
    param_.FromString(param);
}

ChatGlm6BEncoderRopeModel::~ChatGlm6BEncoderRopeModel() {}

uint32_t ChatGlm6BEncoderRopeModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t ChatGlm6BEncoderRopeModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status ChatGlm6BEncoderRopeModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                  std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const size_t dim2 = 2;
    const size_t dim3 = 3;
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = dim3;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[2] = graph_.weightTensors.at(0).desc.shape.dims[1];
    outTensorDescs.at(1) = outTensorDescs.at(0);
    outTensorDescs.at(1).shape.dimNum += 1;
    outTensorDescs.at(1).shape.dims[dim2] = param_.headNum;
    outTensorDescs.at(1).shape.dims[dim3] = param_.dk;
    for (size_t i = 2; i < GetOutputNum(); i++) {
        outTensorDescs.at(i) = outTensorDescs.at(1);
    }
    return atb::NO_ERROR;
}

void ChatGlm6BEncoderRopeModel::BuildGraph()
{
#if 0
    const int weightTensorSize =
        WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(1 + 2 * param_.layerNum);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer_old::EmbeddingParam embeddingParam;
    atb::Operation *op = nullptr;
    atb::CreateOp(embeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    auto &transposeNode = graph_.nodes.at(nodeId++);
    atb::infer::TransposeParam transposeParam = {{1, 0, 2}};
    atb::CreateOp(transposeParam, &op);
    transposeNode.operation.reset(op);
    transposeNode.inTensors = {&graph_.internalTensors.at(0)};
    transposeNode.outTensors = {&graph_.internalTensors.at(1)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(1);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        ChatGlm6BLayerEncoderParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.transKey = param_.transKey;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.residualAddScale = param_.residualAddScale;
        CreateChatGlm6BLayerEncoderOperation(opParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);      // cosTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);      // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);        // seqLen

        layerNode.outTensors = {&graph_.internalTensors.at(OPERATION_COUNT_BEFORE_LAYER + layerId),
                                &graph_.outTensors.at(1 + layerId),
                                &graph_.outTensors.at(1 + param_.layerNum + layerId)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer_old::NormParam finalNormParam;
    finalNormParam.layerNormEps = param_.layerNormEps;
    atb::CreateOp(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT;
    const int finalLayerNormBiasTensorId = graph_.weightTensors.size() - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId),
                               &graph_.weightTensors.at(finalLayerNormBiasTensorId)};
    finalNormNode.outTensors = {&graph_.outTensors.at(0)};
#endif
}
} // namespace atb_speed