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
#include "chatglm2_6b_encoder_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "models/chatglm2_6b/chatglm2_6b_fusion_layer_encoder_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/lm_head_slice_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 7;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int WEIGHT_COUNT_AFTER_LAYER = 2;
const int OPERATION_COUNT_BEFORE_LAYER = 2;
const int OPERATION_COUNT_AFTER_LAYER = 4;

enum InTensorId {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_ROPECACHE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX,
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void ChatGlm2EncoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    transKey = paramJson["transKey"].get<bool>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int>();
    numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int>();
    seqLen = paramJson["seqLen"].get<int>();
   

    ASD_LOG(INFO) << "ChatGlm2EncoderModel param rmsNormEps:" << rmsNormEps 
                  << ", transKey:" << transKey <<  ", layerNum:" << layerNum
                  << ", residualAddScale:" << residualAddScale  << ", numHeadsPerPartition: " << numHeadsPerPartition
                  << ", hiddenSizePerHead:" << hiddenSizePerHead << ", numGroupsPerPartition:" << numGroupsPerPartition;
}

ChatGlm2EncoderModel::ChatGlm2EncoderModel(const std::string &param) : Model("ChatGlm2EncoderModel", param)
{
    param_.FromString(param);
}

ChatGlm2EncoderModel::~ChatGlm2EncoderModel() {}

uint64_t ChatGlm2EncoderModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t ChatGlm2EncoderModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status ChatGlm2EncoderModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }
    const int lastWeightTensorId = graph_.weightTensors.size() - 1;
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], 1,
                                 graph_.weightTensors.at(lastWeightTensorId).desc.dims[0]};
    
    outTensorDescs.at(1) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(1).dims = {inTensors.at(0).desc.dims[1], inTensors.at(0).desc.dims[0],
                                 param_.numGroupsPerPartition, param_.hiddenSizePerHead};

    for (size_t i = 2; i < outTensorDescs.size(); ++i) {
        outTensorDescs.at(i) = outTensorDescs.at(1);
    }
    return AsdOps::Status::OkStatus();
}

void ChatGlm2EncoderModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum + WEIGHT_COUNT_AFTER_LAYER + WORDEMBEDDINGNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    // param_.layerNum * 2 (one for pastK, one for pastV)
    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX + param_.layerNum * 2);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    wordEmbeddingNode.operation = std::make_shared<EmbeddingOperation>(EmbeddingParam());
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    auto &transposeNode = graph_.nodes.at(nodeId++);
    TransposeParam transposeParam = {{1, 0, 2}};
    transposeNode.operation = std::make_shared<TransposeOperation>(transposeParam);
    transposeNode.inTensors = {&graph_.internalTensors.at(0)};
    transposeNode.outTensors = {&graph_.internalTensors.at(1)};

    AsdOps::Tensor *firstInTensor = &graph_.internalTensors.at(1);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        ChatGlm2LayerParam opParam;
        opParam.numHeadsPerPartition = param_.numHeadsPerPartition;
        opParam.numGroupsPerPartition = param_.numGroupsPerPartition;
        opParam.hiddenSizePerHead = param_.hiddenSizePerHead;
        opParam.layerId = layerId;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.preScale = layerId + 1;
        opParam.postScale = layerId + 1;
        opParam.transKey = param_.transKey;
        opParam.model = "chatglm2_6b";
        
        layerNode.operation = std::make_shared<ChatGlm2FusionLayerEncoderOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
        layerNode.outTensors.resize(layerNode.operation->GetOutTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = 
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ROPECACHE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);

        layerNode.outTensors = {&graph_.internalTensors.at(OPERATION_COUNT_BEFORE_LAYER + layerId), &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        firstInTensor = layerNode.outTensors.at(0);
    }

    int internalTensorId = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum;

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    RmsNormParam finalNormParam = {param_.rmsNormEps};
    finalNormNode.operation = std::make_shared<RmsNormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - WEIGHT_COUNT_AFTER_LAYER;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(internalTensorId)};

    auto &sliceNode = graph_.nodes.at(nodeId++);
    LmHeadSliceParam sliceParam = {param_.seqLen};
    sliceNode.operation = std::make_shared<LmHeadSliceOperation>(sliceParam);
    sliceNode.inTensors = {&graph_.internalTensors.at(internalTensorId++)};
    sliceNode.outTensors = {&graph_.internalTensors.at(internalTensorId)};

    auto &lmNode = graph_.nodes.at(nodeId++);
    LinearParam lmParam;
    lmParam.hasBias = false;
    lmNode.operation = std::make_shared<LinearOperation>(lmParam);
    const int lastWeightTensorId = graph_.weightTensors.size() - 1;
    lmNode.inTensors = {&graph_.internalTensors.at(internalTensorId++), &graph_.weightTensors.at(lastWeightTensorId)};
    lmNode.outTensors = {&graph_.internalTensors.at(internalTensorId)};

    auto &transposeNode1 = graph_.nodes.at(nodeId++);
    transposeNode1.operation = std::make_shared<TransposeOperation>(transposeParam);
    transposeNode1.inTensors = {&graph_.internalTensors.at(internalTensorId)};
    transposeNode1.outTensors = {&graph_.outTensors.at(0)};
}

AsdOps::Status ChatGlm2EncoderModel::ParseVarintPackParam(const std::string &param, int nodeId, AsdOps::Any &variantPackParam)
{
    if (nodeId != 31) {
        return AsdOps::Status::OkStatus();
    }
    LmHeadSliceParam opParam;
    nlohmann::json paramJson = nlohmann::json::parse(param);
    opParam.seqLen = paramJson["seqLen"].get<int>();

    ASD_LOG(INFO) << "ChatGlm2EncoderModel::ParseVarintPackParam seqLen: " << opParam.seqLen;

    variantPackParam = opParam;

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
