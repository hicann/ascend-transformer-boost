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
#include "baichuan1_7b_decoder_model.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "models/gptneox20b/gptneox20blayer_embedding_operation.h"
#include "models/baichuan1_7b/baichuan1_7b_layer_decoder_operation.h"

namespace AclTransformer {
const int WEIGHT_COUNT_PER_LAYER = 7;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 1;

enum InTensorId {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PASTK_V_START
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void BaiChuan17BDecoderModel::Param::FromString(const std::string &param)
{
   nlohmann::json paramJson = nlohmann::json::parse(param);
   rmsNormEps = paramJson["rmsNormEps"].get<double>();
   headNum = paramJson["headNum"].get<int>();
   dk = paramJson["dk"].get<int>();
   layerNum = paramJson["layerNum"].get<int>();
   ASD_LOG(INFO) << "Baichuan1_7BDecoderModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                 << ", dk:" << dk << ", layerNum:" << layerNum;
}

BaiChuan17BDecoderModel::BaiChuan17BDecoderModel(const std::string &param): Model("BaiChuan17BDecoderModel", param)
{
   param_.FromString(param);
}

BaiChuan17BDecoderModel::~BaiChuan17BDecoderModel(){}

uint64_t BaiChuan17BDecoderModel::GetInTensorCount() const { return graph_.inTensors.size(); }

uint64_t BaiChuan17BDecoderModel::GetOutTensorCount() const { return graph_.outTensors.size(); }

AsdOps::Status BaiChuan17BDecoderModel::InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                                 std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
   if (outTensorDescs.size() != GetOutTensorCount()) {
       return AsdOps::Status::FailStatus(1, "outTensorDescs size not equal graph outTensors size");
   }

   outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
   outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1],
                                   param_.dk * param_.headNum};

   const AsdOps::Tensor &keyTensor = inTensors.at(IN_TENSOR_PASTK_V_START);
   const AsdOps::Tensor &valueTensor = inTensors.at(IN_TENSOR_PASTK_V_START + param_.layerNum);

   for (size_t keyId = 0; keyId < param_.layerNum; ++keyId) {
       outTensorDescs.at(OUT_TENSOR_MAX + keyId) = keyTensor.desc;
       outTensorDescs.at(OUT_TENSOR_MAX + keyId).dims.at(1) += 1;
   }
   for (size_t valueId = 0; valueId < param_.layerNum; ++valueId) {
       outTensorDescs.at(OUT_TENSOR_MAX + param_.layerNum + valueId) = valueTensor.desc;
       outTensorDescs.at(OUT_TENSOR_MAX + param_.layerNum + valueId).dims.at(1) += 1;
   }

   return AsdOps::Status::OkStatus();
}

void BaiChuan17BDecoderModel::BuildGraph()
{
   const int weightTensorSize =
       WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum + FINALNORMNODE_WEIGHT_COUNT;
   graph_.weightTensors.resize(weightTensorSize);

   graph_.inTensors.resize(IN_TENSOR_PASTK_V_START + 2 * param_.layerNum);
   graph_.outTensors.resize(OUT_TENSOR_MAX);

   const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
   ASD_LOG(INFO) << "BaiChuan1_7BDecoderModel nodeSize is " << nodeSize;
   graph_.nodes.resize(nodeSize);

   graph_.internalTensors.resize(graph_.nodes.size() + 1);

   int nodeId = 0;
   auto &embeddingNode = graph_.nodes.at(nodeId++);
   embeddingNode.operation = std::make_shared<GptNeox20BLayerEmbeddingOperation>(GptNeox20BLayerEmbeddingParam());
   embeddingNode.inTensors.resize(embeddingNode.operation->GetInTensorCount());
   embeddingNode.outTensors.resize(embeddingNode.operation->GetOutTensorCount());
   embeddingNode.inTensors = {&graph_.weightTensors.at(0),
                              &graph_.inTensors.at(IN_TENSOR_INPUTIDS),
                              &graph_.inTensors.at(IN_TENSOR_COSTABLE),
                              &graph_.inTensors.at(IN_TENSOR_SINTABLE),
                              &graph_.inTensors.at(IN_TENSOR_POSITIONID)
   };
   embeddingNode.outTensors = {&graph_.internalTensors.at(0),
                               &graph_.internalTensors.at(1),
                               &graph_.internalTensors.at(2)
   };

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
       layerNode.operation = std::make_shared<BaiChuan17BLayerDecoderOperation>(opParam);
       layerNode.inTensors.resize(layerNode.operation->GetInTensorCount());

       size_t inTensorId = 0;
       layerNode.inTensors.at(inTensorId++) = firstInTensor;
       for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
           layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
               layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
       }
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
       layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;      // cosEmbed
       layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;      // sinEmbed
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor

       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTK_V_START + layerId);
       layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PASTK_V_START + param_.layerNum + layerId);

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

AsdOps::Status BaiChuan17BDecoderModel::ParseVarintPackParam(const std::string &param, int nodeId,
                                                           AsdOps::Any &variantPackParam)
{
   return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer