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
#include "glm130b_decoder_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "models/glm130b/glm130blayer_decoder_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 12;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_MAX,
};

void Glm130BDecoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    transKey = paramJson["transKey"].get<bool>();
    layerNum = paramJson["layerNum"].get<int>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    backend = paramJson["backend"].get<std::string>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    layerNormEps = paramJson["layerNormEps"].get<double>();
    ASD_LOG(INFO) << "Glm130BDecoderModel param layerNormEps:" << layerNormEps << ", headNum:" << headNum
                  << ", layerNum:" << layerNum << ", transKey:" << transKey << ", dk:" << dk
                  << ", residualAddScale:" << residualAddScale << ", rank:" << rank << ", rankSize:" << rankSize
                  << ", backend:" << backend;
}

Glm130BDecoderModel::Glm130BDecoderModel(const std::string &param) : Model("Glm130BDecoderModel", param)
{
    param_.FromString(param);
}

Glm130BDecoderModel::~Glm130BDecoderModel() {}

uint64_t Glm130BDecoderModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t Glm130BDecoderModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status Glm130BDecoderModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                               std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = inTensors.at(0).desc;
    const AsdOps::Tensor &keyTensor = inTensors.at(IN_TENSOR_MAX);
    for (int i = 1; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i) = keyTensor.desc;
        outTensorDescs.at(i).dims.at(0) += 1;
    }

    return AsdOps::Status::OkStatus();
}

void Glm130BDecoderModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(5 + param_.layerNum * 2);
    graph_.outTensors.resize(1 + param_.layerNum * 2);

    const int nodeSize = param_.layerNum;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;

    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(IN_TENSOR_HIDDENSTATES);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        Glm130BLayerParam opParam;
        opParam.transKey = param_.transKey;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;
        opParam.layerId = layerId;
        opParam.rank = param_.rank;
        opParam.rankSize = param_.rankSize;
        opParam.backend = param_.backend;
        opParam.residualAddScale = param_.residualAddScale;
        opParam.layerNormEps = param_.layerNormEps;

        layerNode.operation = std::make_shared<Glm130BLayerDecoderOperation>(opParam);
        layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());
        layerNode.outTensors.resize(layerNode.operation->GetOutTensorCount());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId + param_.layerNum);

        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(layerId), &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0), &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        }

        firstInTensor = layerNode.outTensors.at(0);
    }
}
} // namespace AclTransformer