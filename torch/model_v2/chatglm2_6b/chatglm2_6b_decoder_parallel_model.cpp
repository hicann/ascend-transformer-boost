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
#include "chatglm2_6b_decoder_parallel_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/rms_norm_operation.h"
#include "models/chatglm2_6b/chatglm2_6b_fusion_layer_decoder_parallel_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 7;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_AFTER_LAYER = 1;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_ROPEEMB,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX,
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void ChatGlm2DecoderParallelModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int>();
    numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    
    ASD_LOG(INFO) << "ChatGlm2DecoderParallelModel param rmsNormEps:" << rmsNormEps
                  << ", numHeadsPerPartition:" << numHeadsPerPartition  
                  << ", hiddenSizePerHead:" << hiddenSizePerHead
                  << ", numGroupsPerPartition:" << numGroupsPerPartition
                  << ", transKey:" << transKey  << ", layerNum:" << layerNum
                  << ", residualAddScale:" << residualAddScale;
}

ChatGlm2DecoderParallelModel::ChatGlm2DecoderParallelModel(const std::string &param) : Model("ChatGlm2DecoderParallelModel", param)
{
    param_.FromString(param);
}

ChatGlm2DecoderParallelModel::~ChatGlm2DecoderParallelModel() {}

uint64_t ChatGlm2DecoderParallelModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t ChatGlm2DecoderParallelModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status ChatGlm2DecoderParallelModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = inTensors.at(0).desc;
    const AsdOps::Tensor &keyTensor = inTensors.at(IN_TENSOR_MAX);
    for (size_t i = 1; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i) = keyTensor.desc;
        outTensorDescs.at(i).dims.at(0) += 1;
    }
    return AsdOps::Status::OkStatus();
}

void ChatGlm2DecoderParallelModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum * 2);
    graph_.outTensors.resize(OUT_TENSOR_MAX + param_.layerNum * 2);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(0);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        ChatGlm2LayerParam opParam;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.numHeadsPerPartition = param_.numHeadsPerPartition;
        opParam.hiddenSizePerHead = param_.hiddenSizePerHead;
        opParam.numGroupsPerPartition = param_.numGroupsPerPartition;
        opParam.transKey = param_.transKey;
        opParam.layerId = layerId;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.preScale = layerId + 1;
        opParam.postScale = layerId + 1;
        opParam.model = "chatglm2_6b_parallel";
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;

        layerNode.operation = std::make_shared<ChatGlm2FusionLayerDecoderParallelOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
        layerNode.outTensors.resize(layerNode.operation->GetInTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ROPEEMB);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId + param_.layerNum);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);
        layerNode.outTensors = {&graph_.internalTensors.at(layerId), &graph_.outTensors.at(layerId + 1),
                                &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        firstInTensor = layerNode.outTensors.at(0); 
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    RmsNormParam finalNormParam = {param_.rmsNormEps};
    finalNormNode.operation = std::make_shared<RmsNormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.outTensors.at(0)};
}
} // namespace AclTransformer