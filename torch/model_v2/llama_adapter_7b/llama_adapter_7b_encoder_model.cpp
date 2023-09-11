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
#include "llama_adapter_7b_encoder_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/apply_rotary_emb_operation.h"
#include "acltransformer/ops/self_attention_cross_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "models/llama_adapter7b/llama_adapter_7b_layer_encoder_operation.h"
#include "models/llama_adapter7b/llama_adapter_7b_layer_encoder_adapter_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 15;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_FREQCIS,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_ADAPTER,
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void LlamaAdapter7BEncoderModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    ASD_LOG(INFO) << "Llama_Adapter_7B_Encoder_Model param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                    << ", dk:" << dk << ", layerNum:" << layerNum;
}

LlamaAdapter7BEncoderModel::LlamaAdapter7BEncoderModel(const std::string &param): Model("LlamaAdapter7BEncoderModel", param)
{
    param_.FromString(param);
}

LlamaAdapter7BEncoderModel::~LlamaAdapter7BEncoderModel(){}

uint64_t LlamaAdapter7BEncoderModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t LlamaAdapter7BEncoderModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status LlamaAdapter7BEncoderModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutTensorCount()) {
        return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
    }

    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1], inTensors.at(0).desc.dims[2]};

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

void LlamaAdapter7BEncoderModel::BuildGraph()
{
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_ADAPTER + param_.layerNum - 1);
    graph_.outTensors.resize(OUT_TENSOR_MAX + 2 * param_.layerNum);

    const int nodeSize = param_.layerNum;
    ASD_LOG(INFO) << "Llama_Adapter_7B_Encoder_Model nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size();
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;

    AsdOps::Tensor *firstInTensor = &graph_.inTensors.at(0);;

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        LLaMAAdapter7BLayerParam opParam;
        opParam.rmsNormEps = param_.rmsNormEps;
        opParam.headNum = param_.headNum;
        opParam.dk = param_.dk;

        if (layerId == 0){
            opParam.model = "llama_adapter_encoder";
            layerNode.operation = std::make_shared<LLaMAAdapter7BLayerEncoderOperation>(opParam);
            layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER - 1; ++weightTensorId) {
                if (weightTensorId < 5){
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightTensorId);
                } else{
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(weightTensorId + 1);
                }
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_FREQCIS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            
            layerNode.outTensors = {&graph_.internalTensors.at(layerId),
                                    &graph_.outTensors.at(OUT_TENSOR_MAX + layerId),
                                    &graph_.outTensors.at(OUT_TENSOR_MAX + layerId + param_.layerNum)};
        } else{
            opParam.model = "llama_adapter_encoder_a";
            layerNode.operation = std::make_shared<LLaMAAdapter7BLayerEncoderAdapterOperation>(opParam);
            layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                    layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_FREQCIS);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ADAPTER + layerId - 1);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);

            layerNode.outTensors = {&graph_.internalTensors.at(layerId),
                                    &graph_.outTensors.at(OUT_TENSOR_MAX + layerId),
                                    &graph_.outTensors.at(OUT_TENSOR_MAX + layerId + param_.layerNum)};
        }
            
        firstInTensor = layerNode.outTensors.at(0);

        if (layerId == param_.layerNum - 1){
            layerNode.outTensors.at(0) = {&graph_.outTensors.at(0)};
        }
    }
}

AsdOps::Status LlamaAdapter7BEncoderModel::ParseVarintPackParam(const std::string &param, int nodeId,
                                                           AsdOps::Any &variantPackParam)
{
   return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer