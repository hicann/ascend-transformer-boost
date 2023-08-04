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
#include "operation_creator.h"
#include <nlohmann/json.hpp>
#include <functional>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/linear_parallel_operation.h"
#include "acltransformer/ops/all_reduce_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/post_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/matmul_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/any_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/quant_operation.h"
#include "acltransformer/ops/add_norm_quant_operation.h"
#include "acltransformer/ops/norm_quant_operation.h"
#include "acltransformer/ops/linear_quant_operation.h"
#include "acltransformer/ops/ffn_quant_operation.h"
#include "acltransformer/ops/ffn_quant_operation.h"
#include "acltransformer/ops/lm_head_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_without_fusion_operation.h"
#include "models/chatglm6b/chatglm6blayer_encoder_operation.h"
#include "models/bert/bertlayer_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_quant_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_first_quant_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_last_quant_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_flashattention_operation.h"
#include "models/glm130b/glm130blayer_decoder_operation.h"
#include "models/glm130b/glm130blayer_encoder_operation.h"
#include "models/llama7b/llama7blayer_operation.h"
#include "models/llama13b/llama13blayer_parallel_operation.h"

using OperationCreateFunc = std::function<AclTransformer::Operation *(const nlohmann::json &paramJson)>;

static AclTransformer::Operation *LLaMA7BLayerOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMA7BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    ASD_LOG(INFO) << "LLaMA7BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    return new AclTransformer::LLaMA7BLayerOperation(param);
}

static AclTransformer::Operation *LLaMA13BLayerOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMA13BLayerParam param;
    if (paramJson.find("rmsNormEps") != paramJson.end()) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }
    if (paramJson.find("headNum") != paramJson.end()) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.find("dk") != paramJson.end()) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.find("rank") != paramJson.end()) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.find("rankSize") != paramJson.end()) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ASD_LOG(INFO) << "LLaMA13BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    return new AclTransformer::LLaMA13BLayerOperation(param);
}

static AclTransformer::Operation *PostOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::PostParam param;
    param.top_p = paramJson["top_p"].get<float>();
    param.top_k = paramJson["top_k"].get<int>();
    param.temperature = paramJson["temperature"].get<float>();
    if (paramJson.find("filter_value") != paramJson.end()) {
        param.filter_value = paramJson["filter_value"].get<float>();
    }
    param.min_tokens_to_keep = paramJson["min_tokens_to_keep"].get<int>();
    ASD_LOG(INFO) << "PostParam top_p:" << param.top_p;
    ASD_LOG(INFO) << "PostParam top_k:" << param.top_k;
    ASD_LOG(INFO) << "PostParam temperature:" << param.temperature;
    ASD_LOG(INFO) << "PostParam min_tokens_to_keep:" << param.min_tokens_to_keep;
    return new AclTransformer::PostOperation(param);
}

static AclTransformer::Operation *AllReduceOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AllReduceParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    ASD_LOG(INFO) << "AllReduceParam rank:" << param.rank;
    ASD_LOG(INFO) << "AllReduceParam rankSize:" << param.rankSize;
    return new AclTransformer::AllReduceOperation(param);
}

static AclTransformer::Operation *LinearParallelOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LinearParallelParam param;
    if (paramJson.find("transWeight") != paramJson.end()) {
        param.transWeight = paramJson["transWeight"].get<bool>();
    }
    if (paramJson.find("bias") != paramJson.end()) {
        param.bias = paramJson["bias"].get<std::string>();
    }
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.parallelType = paramJson["parallelType"].get<std::string>();
    return new AclTransformer::LinearParallelOperation(param);
}

static AclTransformer::Operation *AddOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AddParam param;
    if (paramJson.find("scale") != paramJson.end()) {
        param.scale = paramJson["scale"].get<float>();
    }
    ASD_LOG(INFO) << "AddParam scale:" << param.scale;
    return new AclTransformer::AddOperation(param);
}
AclTransformer::Operation *RopeOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::PositionEmbeddingFusionParam param;
    param.headNum = paramJson["headNum"].get<int64_t>();
    ASD_LOG(INFO) << "param.headNum: " << param.headNum;
    return new AclTransformer::RopeOperation(param);
}
static AclTransformer::Operation *AddNormOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AddNormParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    ASD_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps;
    param.zoom_scale = paramJson["zoom_scale"].get<float>();
    ASD_LOG(INFO) << "NormParam zoom_scale:" << param.zoom_scale;
    return new AclTransformer::AddNormOperation(param);
}

static AclTransformer::Operation *RmsNormOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::RmsNormParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<double>();
    return new AclTransformer::RmsNormOperation(param);
}

static AclTransformer::Operation *EmbeddingOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::EmbeddingParam param;
    ASD_LOG(INFO) << "EmbeddingParam axis:" << param.axis;
    return new AclTransformer::EmbeddingOperation(param);
}

static AclTransformer::Operation *NormOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::NormParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    if (paramJson.contains("beginNormAxis")) {
        param.beginNormAxis = paramJson["beginNormAxis"].get<int32_t>();
    }
    if (paramJson.contains("beginParamsAxis")) {
        param.beginParamsAxis = paramJson["beginParamsAxis"].get<int32_t>();
    }
    ASD_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps << ", beginNormAxis:" << param.beginNormAxis
                  << ", beginParamsAxis:" << param.beginParamsAxis;
    return new AclTransformer::NormOperation(param);
}

static AclTransformer::Operation *LinearOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LinearParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    ASD_LOG(INFO) << "LinearParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias;
    return new AclTransformer::LinearOperation(param);
}

static AclTransformer::Operation *MatmulOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::MatmulParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    ASD_LOG(INFO) << "MatmulParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB;
    return new AclTransformer::MatmulOperation(param);
}

static AclTransformer::Operation *FfnOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::FfnParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    ASD_LOG(INFO) << "FfnParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias;
    return new AclTransformer::FfnOperation(param);
}

static AclTransformer::Operation *MlpOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::MlpParam param;
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
        ASD_LOG(INFO) << "MlpParam model:" << param.model;
    } else {
        param.model = "llama7b";
        ASD_LOG(INFO) << "MlpParam is empty, default model:" << param.model;
    }
    return new AclTransformer::MlpOperation(param);
}

static AclTransformer::Operation *AnyOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AnyParam param;
    param.kernelGraph = paramJson;
    return new AclTransformer::AnyOperation(param);
}

static AclTransformer::Operation *SelfAttentionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::SelfAttentionParam param;
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    ASD_LOG(INFO) << "SelfAttentionKvCacheParam transKey:" << param.transKey << ", headNum:" << param.headNum
                  << ", layerId:" << param.layerId << ", dk:" << param.dk;
    return new AclTransformer::SelfAttentionOperation(param);
}

static AclTransformer::Operation *PositionEmbedding1dSplitOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::PositionEmbedding1dSplitParam param;
    param.headNum = paramJson["headNum"].get<int>();
    ASD_LOG(INFO) << "PositionEmbeddingParam headNum:" << param.headNum;
    return new AclTransformer::PositionEmbedding1dSplitOperation(param);
}

static AclTransformer::Operation *PositionEmbeddingOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::PositionEmbeddingParam param;
    param.headNum = paramJson["headNum"].get<int>();
    ASD_LOG(INFO) << "PositionEmbeddingParam headNum:" << param.headNum;
    return new AclTransformer::PositionEmbeddingOperation(param);
}

static AclTransformer::Operation *SelfAttentionKvCacheOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    ASD_LOG(INFO) << "SelfAttentionKvCacheParam transKey:" << param.transKey << ", headNum:" << param.headNum
                  << ", layerId:" << param.layerId << ", dk:" << param.dk;
    return new AclTransformer::SelfAttentionKvCacheOperation(param);
}

static AclTransformer::Operation *TransposeOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::TransposeParam param;
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    ASD_LOG(INFO) << "transpose(" << param.perm << ")";
    return new AclTransformer::TransposeOperation(param);
}

static AclTransformer::Operation *ChatGlm6BLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BLayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    ASD_LOG(INFO) << "ChatGlm6BLayerDecoderParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale;
    return new AclTransformer::ChatGlm6BLayerDecoderOperation(param);
}

static AclTransformer::Operation *ChatGlm6BLayerDecoderWithoutFusionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BLayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.beginNormAxis = paramJson["beginNormAxis"].get<int>();
    ASD_LOG(INFO) << "ChatGlm6BLayerDecoderParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale << ", beginNormAxis:" << param.beginNormAxis;
    return new AclTransformer::ChatGlm6BLayerDecoderWithoutFusionOperation(param);
}

static AclTransformer::Operation *ChatGlm6BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BLayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    ASD_LOG(INFO) << "ChatGlm6BLayerEncoderParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale;
    return new AclTransformer::ChatGlm6BLayerEncoderOperation(param);
}

AclTransformer::Operation *FfnQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::FfnQuantParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    ASD_LOG(INFO) << "FfnParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB;
    return new AclTransformer::FfnQuantOperation(param);
}

AclTransformer::Operation *LinearQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LinearQuantParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    ASD_LOG(INFO) << "LinearQuantParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB;
    return new AclTransformer::LinearQuantOperation(param);
}

AclTransformer::Operation *AddNormQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AddNormQuantParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.inputScale = paramJson["input_scale"].get<float>();
    param.inputOffset = paramJson["input_offset"].get<int>();
    param.inputAlpha = paramJson["input_alpha"].get<float>();

    ASD_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps << ", input_scale:" << param.inputScale
                  << ", input_offset:" << param.inputOffset << ", input_alpha:" << param.inputAlpha;
    return new AclTransformer::AddNormQuantOperation(param);
}

AclTransformer::Operation *NormQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::NormQuantParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.inputScale = paramJson["input_scale"].get<float>();
    param.inputOffset = paramJson["input_offset"].get<int>();
    param.inputAlpha = paramJson["input_alpha"].get<float>();

    ASD_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps << ", input_scale:" << param.inputScale
                  << ", input_offset:" << param.inputOffset << ", input_alpha:" << param.inputAlpha;
    return new AclTransformer::NormQuantOperation(param);
}

AclTransformer::Operation *QuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::QuantParam param;
    param.inputScale = paramJson["input_scale"].get<float>();
    param.inputOffset = paramJson["input_offset"].get<int>();
    ASD_LOG(INFO) << "QuantParam input scale:" << param.inputScale << ", input_offset:" << param.inputOffset;
    return new AclTransformer::QuantOperation(param);
}

AclTransformer::Operation *SelfAttentionKvCacheFusionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::SelfAttentionKvCacheFusionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
        ASD_LOG(FATAL) << "token offset:" << param.tokenOffset.at(0);
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
        ASD_LOG(FATAL) << "seqLen:" << param.seqLen.at(0);
    }
    ASD_LOG(INFO) << "SelfAttentionKvCacheFusionParam headNum:" << param.headNum;
    AclTransformer::Operation *opAddr = new AclTransformer::SelfAttentionKvCacheFusionOperation(param);
    ASD_LOG(FATAL) << "SelfAttentionKvCacheFusionOperation addr:" << opAddr;
    return opAddr;
}

AclTransformer::Operation *BertLayerOperation(const nlohmann::json &paramJson)
{
    AclTransformer::BertLayerParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    return new AclTransformer::BertLayerOperation(param);
}

static AclTransformer::Operation *ChatGlm6BLayerDecoderQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BLayerQuantParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.qkvInputScale = paramJson["qkvInputScale"].get<float>();
    param.qkvInputOffset = paramJson["qkvInputOffset"].get<int>();
    param.denseInputScale = paramJson["denseInputScale"].get<float>();
    param.denseInputOffset = paramJson["denseInputOffset"].get<int>();
    param.selfLnInputScale = paramJson["selfLnInputScale"].get<float>();
    param.selfLnInputOffset = paramJson["selfLnInputOffset"].get<int>();
    param.ffnOutInputScale = paramJson["ffnOutInputScale"].get<float>();
    param.ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<int>();

    ASD_LOG(INFO) << "ChatGlm6BLayerParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale << ", qkvInputScale:" << param.qkvInputScale
                  << ", qkvInputOffset" << param.qkvInputOffset << ", denseInputScale" << param.denseInputScale
                  << ", denseInputOffset" << param.denseInputOffset << ", selfLnInputScale" << param.selfLnInputScale
                  << ", selfLnInputOffset" << param.selfLnInputOffset << ", ffnOutInputScale" << param.ffnOutInputScale
                  << ", ffnOutInputOffset" << param.ffnOutInputOffset;
    return new AclTransformer::ChatGlm6BLayerDecoderQuantOperation(param);
}

static AclTransformer::Operation *ChatGlm6BLayerDecoderFirstQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BLayerQuantParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.qkvInputScale = paramJson["qkvInputScale"].get<float>();
    param.qkvInputOffset = paramJson["qkvInputOffset"].get<int>();
    param.denseInputScale = paramJson["denseInputScale"].get<float>();
    param.denseInputOffset = paramJson["denseInputOffset"].get<int>();
    param.selfLnInputScale = paramJson["selfLnInputScale"].get<float>();
    param.selfLnInputOffset = paramJson["selfLnInputOffset"].get<int>();
    param.ffnOutInputScale = paramJson["ffnOutInputScale"].get<float>();
    param.ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<int>();

    ASD_LOG(INFO) << "ChatGlm6BLayerParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale << ", qkvInputScale:" << param.qkvInputScale
                  << ", qkvInputOffset" << param.qkvInputOffset << ", denseInputScale" << param.denseInputScale
                  << ", denseInputOffset" << param.denseInputOffset << ", selfLnInputScale" << param.selfLnInputScale
                  << ", selfLnInputOffset" << param.selfLnInputOffset << ", ffnOutInputScale" << param.ffnOutInputScale
                  << ", ffnOutInputOffset" << param.ffnOutInputOffset;
    return new AclTransformer::ChatGlm6BLayerDecoderFirstQuantOperation(param);
}

static AclTransformer::Operation *ChatGlm6BLayerDecoderLastQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BLayerQuantParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.qkvInputScale = paramJson["qkvInputScale"].get<float>();
    param.qkvInputOffset = paramJson["qkvInputOffset"].get<int>();
    param.denseInputScale = paramJson["denseInputScale"].get<float>();
    param.denseInputOffset = paramJson["denseInputOffset"].get<int>();
    param.selfLnInputScale = paramJson["selfLnInputScale"].get<float>();
    param.selfLnInputOffset = paramJson["selfLnInputOffset"].get<int>();
    param.ffnOutInputScale = paramJson["ffnOutInputScale"].get<float>();
    param.ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<int>();

    ASD_LOG(INFO) << "ChatGlm6BLayerParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale << ", qkvInputScale:" << param.qkvInputScale
                  << ", qkvInputOffset" << param.qkvInputOffset << ", denseInputScale" << param.denseInputScale
                  << ", denseInputOffset" << param.denseInputOffset << ", selfLnInputScale" << param.selfLnInputScale
                  << ", selfLnInputOffset" << param.selfLnInputOffset << ", ffnOutInputScale" << param.ffnOutInputScale
                  << ", ffnOutInputOffset" << param.ffnOutInputOffset;
    return new AclTransformer::ChatGlm6BLayerDecoderLastQuantOperation(param);
}

static AclTransformer::Operation *ChatGlm6BLayeEncoderFlashAttentionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BLayerDecoderFlashAttentionParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    return new AclTransformer::ChatGlm6BLayerDecoderFlashAttentionOperation(param);
}

AclTransformer::Operation *Glm130BLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::Glm130BLayerParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("residualAddScale")) {
        param.residualAddScale = paramJson["residualAddScale"].get<float>();
    }
    if (paramJson.contains("layerNormEps")) {
        param.layerNormEps = paramJson["layerNormEps"].get<double>();
    }
    return new AclTransformer::Glm130BLayerDecoderOperation(param);
}

AclTransformer::Operation *Glm130BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::Glm130BLayerParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("residualAddScale")) {
        param.residualAddScale = paramJson["residualAddScale"].get<float>();
    }
    if (paramJson.contains("layerNormEps")) {
        param.layerNormEps = paramJson["layerNormEps"].get<double>();
    }
    return new AclTransformer::Glm130BLayerEncoderOperation(param);
}

AclTransformer::Operation *LmHeadOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LmHeadParam param;
    return new AclTransformer::LmHeadOperation(param);
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"PostOperation", &PostOperationCreate},
    {"AllReduceOperation", AllReduceOperationCreate},
    {"LinearParallelOperation", &LinearParallelOperationCreate},
    {"AddOperation", &AddOperationCreate},
    {"NormOperation", &NormOperationCreate},
    {"RopeOperation", &RopeOperationCreate},
    {"AddNormOperation", &AddNormOperationCreate},
    {"RmsNormOperation", &RmsNormOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"MatmulOperation", &MatmulOperationCreate},
    {"FfnOperation", &FfnOperationCreate},
    {"MlpOperation", &MlpOperationCreate},
    {"EmbeddingOperation", &EmbeddingOperationCreate},
    {"PositionEmbedding1dSplitOperation", &PositionEmbedding1dSplitOperationCreate},
    {"PositionEmbeddingOperation", &PositionEmbeddingOperationCreate},
    {"SelfAttentionKvCacheOperation", &SelfAttentionKvCacheOperationCreate},
    {"SelfAttentionKvCacheFusionOperation", &SelfAttentionKvCacheFusionOperationCreate},
    {"SelfAttentionOperation", &SelfAttentionOperationCreate},
    {"AnyOperation", &AnyOperationCreate},
    {"ChatGlm6BLayerDecoderOperation", &ChatGlm6BLayerDecoderOperationCreate},
    {"ChatGlm6BLayerDecoderWithoutFusionOperation", &ChatGlm6BLayerDecoderWithoutFusionOperationCreate},
    {"ChatGlm6BLayerEncoderOperation", &ChatGlm6BLayerEncoderOperationCreate},
    {"QuantOperation", &QuantOperationCreate},
    {"AddNormQuantOperation", &AddNormQuantOperationCreate},
    {"NormQuantOperation", &NormQuantOperationCreate},
    {"LinearQuantOperation", &LinearQuantOperationCreate},
    {"FfnQuantOperation", &FfnQuantOperationCreate},
    {"BertLayerOperation", &BertLayerOperation},
    {"FfnQuantOperation", &FfnQuantOperationCreate},
    {"ChatGlm6BLayerDecoderQuantOperation", &ChatGlm6BLayerDecoderQuantOperationCreate},
    {"ChatGlm6BLayerDecoderLastQuantOperation", &ChatGlm6BLayerDecoderLastQuantOperationCreate},
    {"ChatGlm6BLayerDecoderFirstQuantOperation", &ChatGlm6BLayerDecoderFirstQuantOperationCreate},
    {"ChatGlm6BLayerDecoderFlashAttentionOperation", &ChatGlm6BLayeEncoderFlashAttentionOperationCreate},
    {"Glm130BLayerDecoderOperation", &Glm130BLayerDecoderOperationCreate},
    {"Glm130BLayerEncoderOperation", &Glm130BLayerEncoderOperationCreate},
    {"LLaMA7BLayerOperation", &LLaMA7BLayerOperationCreate},
    {"LmHeadOperation", &LmHeadOperationCreate},
    {"LLaMA13BLayerOperation", &LLaMA13BLayerOperationCreate}};

AclTransformer::Operation *CreateOperation(const std::string &opName, const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        ASD_LOG(ERROR) << "not support opName:" << opName;
        return nullptr;
    }

    try {
        return it->second(paramJson);
    } catch (const std::exception &e) {
        ASD_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return nullptr;
}

AsdOps::Any ParseParam(const std::string &opName, const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    if (opName == "ChatGlm6BLayerDecoderFlashAttentionOperation") {
        AclTransformer::SelfAttentionKvCacheFusionVariantPackParam opParam;
        for (auto item : paramJson["tokenOffset"]) {
            opParam.tokenOffset.push_back(item.get<int>());
        }
        for (auto item : paramJson["seqLen"]) {
            opParam.seqLen.push_back(item.get<int>());
        }
        opParam.layerId = paramJson["layerId"].get<int>();
        return opParam;
    }
    return AsdOps::Any();
}