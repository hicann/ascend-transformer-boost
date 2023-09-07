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
#include "bloom7b_decoder_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "models/bloom7b/bloom7blayer_param.h"
#include "models/bloom7b/bloom7blayer_decoder_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 12;
const int FINAL_LINEAR_WEIGHT_COUNT = 1;
const int FINAL_NORM_WEIGHT_COUNT = 2;

enum InTensorId { IN_HIDDEN_STATES = 0, IN_ALIBI, IN_ATTENTION_MASK, IN_PAST_KEY };

void Bloom7BDecoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();
    activationFuncType = paramJson["activationFuncType"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    ASD_LOG(INFO) << "Bloom7BDecoderModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", invNormFactorvarAttr:" << invNormFactorvarAttr
                  << ", activationFuncType:" << activationFuncType << ", layerNum" << layerNum;
}

Bloom7BDecoderModel::Bloom7BDecoderModel(const std::string &param) : Model("Bloom7BDecoderModel", param)
{
    param_.FromString(param);
}

Bloom7BDecoderModel::~Bloom7BDecoderModel()
{}

uint64_t Bloom7BDecoderModel::GetInTensorCount() const
{
    return graph_.inTensors.size();
}

uint64_t Bloom7BDecoderModel::GetOutTensorCount() const
{
    return graph_.outTensors.size();
}

AsdOps::Status Bloom7BDecoderModel::InferShape(
    const std::vector<AsdOps::Tensor> &inTensors, std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }
    const AsdOps::Tensor &hiddenState = inTensors.at(IN_HIDDEN_STATES);
    const AsdOps::Tensor &keyTensor = inTensors.at(IN_PAST_KEY);
    const AsdOps::Tensor &valueTensor = inTensors.at(IN_PAST_KEY + param_.layerNum);

    outTensorDescs.at(0) = hiddenState.desc;
    for (size_t keyId = 0; keyId < param_.layerNum; ++keyId) {
        outTensorDescs.at(1 + keyId) = keyTensor.desc;
        outTensorDescs.at(1 + keyId).dims.at(2) += 1;
    }

    for (size_t valueId = 0; valueId < param_.layerNum; ++valueId) {
        outTensorDescs.at(1 + param_.layerNum + valueId) = valueTensor.desc;
        outTensorDescs.at(1 + param_.layerNum + valueId).dims.at(1) += 1;
    }

    outTensorDescs.at(outTensorDescs.size() - 1) = hiddenState.desc;
    outTensorDescs.at(outTensorDescs.size() - 1).dims.clear();
    outTensorDescs.at(outTensorDescs.size() - 1).dims = {hiddenState.desc.dims.at(0),hiddenState.desc.dims.at(1), 250880}; 

    return AsdOps::Status::OkStatus();
}

void Bloom7BDecoderModel::BuildGraph()
{
    ASD_LOG(INFO) << "Build Graph Start.";

    const int weightTensorSize =
        WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINAL_LINEAR_WEIGHT_COUNT + FINAL_NORM_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(2 * param_.layerNum + 3);
    graph_.outTensors.resize(2 * param_.layerNum + 2);

    const int nodeSize = param_.layerNum + 2;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;

    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(0);
    ASD_LOG(INFO) << "First InTensor Set.";

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        Bloom7BLayerParam opParam;
        opParam.layerNormEps = param_.layerNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.invNormFactorvarAttr = param_.invNormFactorvarAttr;
        opParam.activationFuncType = param_.activationFuncType;
        layerNode.operation = std::make_shared<Bloom7BLayerDecoderOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
        ASD_LOG(INFO) << "layerNode Set." << layerId;
        size_t inTensorId = 0;

        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        ASD_LOG(INFO) << "weightTensors Set." << layerId;

        layerNode.inTensors.at(inTensorId++) = firstInTensor;

        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ALIBI);

        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_ATTENTION_MASK);

        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PAST_KEY + layerId);

        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PAST_KEY + param_.layerNum + layerId);

        ASD_LOG(INFO) << "inTensors Set." << layerId;
        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(layerId),
                &graph_.outTensors.at(layerId + 1),
                &graph_.outTensors.at(layerId + param_.layerNum + 1)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0),
                &graph_.outTensors.at(layerId + 1),
                &graph_.outTensors.at(layerId + param_.layerNum + 1)};
        }
        firstInTensor = layerNode.outTensors.at(0);
        ASD_LOG(INFO) << "firstInTensor Set.";
    }
    
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    NormParam finalNormParam;
    finalNormParam.layerNormEps = param_.layerNormEps;
    finalNormNode.operation = std::make_shared<NormOperation>(finalNormParam);
    const int finalLayerNormWeightTensorId = 
        graph_.weightTensors.size() - FINAL_LINEAR_WEIGHT_COUNT - FINAL_NORM_WEIGHT_COUNT;
    finalNormNode.inTensors = {firstInTensor,
        &graph_.weightTensors.at(finalLayerNormWeightTensorId),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId + 1)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(graph_.internalTensors.size() - 1)};

    auto &finalLinearNode = graph_.nodes.at(nodeId++);
    LinearParam finalLinearParam;
    finalLinearParam.hasBias = false;
    finalLinearNode.operation = std::make_shared<LinearOperation>(finalLinearParam);
    const int finalLinearNodeWeightTensorId = graph_.weightTensors.size() - FINAL_LINEAR_WEIGHT_COUNT;
    finalLinearNode.inTensors = {&graph_.internalTensors.at(graph_.internalTensors.size() - 1),
        &graph_.weightTensors.at(finalLinearNodeWeightTensorId)};
    finalLinearNode.outTensors = {&graph_.outTensors.at(graph_.outTensors.size() - 1)};
    ASD_LOG(INFO) << "Build Graph finished.";
}

AsdOps::Status Bloom7BDecoderModel::ParseVarintPackParam(
    const std::string &param, int nodeId, AsdOps::Any &variantPackParam)
{
    return AsdOps::Status::OkStatus();
}
}  // namespace AclTransformer