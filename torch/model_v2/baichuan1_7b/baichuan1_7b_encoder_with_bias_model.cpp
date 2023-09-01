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
#include "baichuan1_7b_encoder_with_bias_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "models/gptneox20b/gptneox20blayer_embedding_operation.h"
#include "models/baichuan1_7b/baichuan1_7b_layer_encoder_with_bias_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 9;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 1;
const int IN_TENSOR_COUNT=6;
const int OUT_BASE_TENSOR_COUNT=1;

enum InTensorId {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_SEQLEN,
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
};

void BaiChuan17BEncoderWithBiasModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    ASD_LOG(INFO) << "GptNeox20BDecoderModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum;
}

BaiChuan17BEncoderWithBiasModel::BaiChuan17BEncoderWithBiasModel(const std::string &param) : Model("BaiChuan17BEncoderWithBiasModel", param)
{
    param_.FromString(param);
}

BaiChuan17BEncoderWithBiasModel::~BaiChuan17BEncoderWithBiasModel() {}

uint64_t BaiChuan17BEncoderWithBiasModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t BaiChuan17BEncoderWithBiasModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status BaiChuan17BEncoderWithBiasModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                   std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1],
                                 param_.headNum * param_.dk};

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

void BaiChuan17BEncoderWithBiasModel::BuildGraph()
{
    const int weightTensorSize =
        WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    //
    graph_.inTensors.resize(IN_TENSOR_COUNT);
    // param_.layerNum * 2 (one for presentK, one for presentV)
    graph_.outTensors.resize(OUT_BASE_TENSOR_COUNT + param_.layerNum * 2);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    ASD_LOG(INFO) << "BaiChuan17BEncoderWithBiasModel nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() + 1);

    int nodeId = 0;
    auto &embeddingNode = graph_.nodes.at(nodeId++);
    embeddingNode.operation = std::make_shared<GptNeox20BLayerEmbeddingOperation>(GptNeox20BLayerEmbeddingParam());
    embeddingNode.inTensors.resize(embeddingNode.operation->GetInTensorCount());
    embeddingNode.outTensors.resize(embeddingNode.operation->GetOutTensorCount());
    embeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUTIDS),
                               &graph_.inTensors.at(IN_TENSOR_COSTABLE), &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                               &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    embeddingNode.outTensors = {&graph_.internalTensors.at(0), &graph_.internalTensors.at(1),
                                &graph_.internalTensors.at(2)};

    AsdOps::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    AsdOps::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    AsdOps::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        BaiChuan17BLayerParam opParam;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.model = "baichuan1_7b";
        layerNode.operation = std::make_shared<BaiChuan17BLayerEncoderWithBiasOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
        layerNode.outTensors.resize(layerNode.operation->GetOutTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;                                // cosEmbed
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;                                // sinEmbed
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor

        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId),
                                &graph_.outTensors.at(1 + layerId),
                                &graph_.outTensors.at(1 + layerId + param_.layerNum)};

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