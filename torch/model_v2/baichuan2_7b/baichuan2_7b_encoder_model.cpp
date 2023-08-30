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
#include "baichuan2_7b_encoder_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "models/gptneox20b/gptneox20blayer_embedding_operation.h"
#include "models/baichuan2_7b/baichuan2_7b_layer_encoder_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 7;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int IN_TENSOR_COUNT=6;
const int OUT_BASE_TENSOR_COUNT=1;

enum InTensorId {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSEMBED,
    IN_TENSOR_SINEMBED,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_SEQLEN,
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
};

void BaiChuan27BEncoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    ASD_LOG(INFO) << "BaiChuan27BEncoderModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum;
}

BaiChuan27BEncoderModel::BaiChuan27BEncoderModel(const std::string &param) : Model("BaiChuan27BEncoderModel", param)
{
    param_.FromString(param);
}

BaiChuan27BEncoderModel::~BaiChuan27BEncoderModel() {}

uint64_t BaiChuan27BEncoderModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t BaiChuan27BEncoderModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status BaiChuan27BEncoderModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                   std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dims[0];
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1],
                                    outDim};

    outTensorDescs.at(1) = outTensorDescs.at(0);
    outTensorDescs.at(1).dims.clear();
    outTensorDescs.at(1).dims.push_back(inTensors.at(0).desc.dims.at(0));
    outTensorDescs.at(1).dims.push_back(inTensors.at(0).desc.dims.at(1));
    outTensorDescs.at(1).dims.push_back(param_.headNum);
    outTensorDescs.at(1).dims.push_back(param_.dk);

    for (size_t i = 2; i < outTensorDescs.size(); ++i) {
        outTensorDescs.at(i) = outTensorDescs.at(1);
    }
    return AsdOps::Status::OkStatus();
}

void BaiChuan27BEncoderModel::BuildGraph()
{
    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT +
                                 WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                 FINALNORMNODE_WEIGHT_COUNT +
                                 OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    //
    graph_.inTensors.resize(IN_TENSOR_COUNT);
    // param_.layerNum * 2 (one for presentK, one for presentV)
    graph_.outTensors.resize(OUT_BASE_TENSOR_COUNT + param_.layerNum * 2);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    ASD_LOG(INFO) << "BaiChuan27BEncoderModel nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    wordEmbeddingNode.operation = std::make_shared<EmbeddingOperation>(EmbeddingParam());
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    AsdOps::Tensor *firstInTensor = &graph_.internalTensors.at(0);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        BaiChuan27BLayerParam opParam;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.model = "baichuan2_7b";
        layerNode.operation = std::make_shared<BaiChuan27BLayerEncoderOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
        layerNode.outTensors.resize(layerNode.operation->GetOutTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSEMBED);      // cosEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINEMBED);      // sinEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor

        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId),
                                &graph_.outTensors.at(1 + layerId),
                                &graph_.outTensors.at(1 + layerId + param_.layerNum)};

        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    RmsNormParam finalNormParam = {param_.rmsNormEps};
    finalNormNode.operation = std::make_shared<RmsNormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT
                                             - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    LinearParam linearParam;
    linearParam.hasBias = false;
    outLinearNode.operation = std::make_shared<LinearOperation>(linearParam);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};
}
} // namespace AclTransformer