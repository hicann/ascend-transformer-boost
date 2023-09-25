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
#include "acltransformer/ops/all_gather_operation.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/post_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/matmul_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/apply_rotary_emb_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/self_attention_cross_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include "acltransformer/ops/position_embedding_1d_fusion_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/any_operation.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "acltransformer/ops/quant_operation.h"
#include "acltransformer/ops/add_norm_quant_operation.h"
#include "acltransformer/ops/norm_quant_operation.h"
#include "acltransformer/ops/rms_norm_quant_operation.h"
#include "acltransformer/ops/rms_pre_norm_quant_operation.h"
#include "acltransformer/ops/mlp_quant_operation.h"
#include "acltransformer/ops/linear_quant_operation.h"
#include "acltransformer/ops/ffn_quant_operation.h"
#include "acltransformer/ops/ffn_quant_operation.h"
#include "acltransformer/ops/lm_head_operation.h"
#include "acltransformer/ops/lm_head_slice_operation.h"
#include "acltransformer/ops/lm_head_parallel_operation.h"
#include "acltransformer/ops/word_embedding_parallel_operation.h"
#include "acltransformer/ops/transdata_int8_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_without_fusion_operation.h"
#include "models/chatglm6b/chatglm6blayer_encoder_operation.h"
#include "models/bert/bertlayer_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_quant_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_first_quant_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_last_quant_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_flashattention_operation.h"
#include "models/glm130b/glm130blayer_decoder_operation.h"
#include "models/glm130b/glm130blayer_decoder_with_fusion_operation.h"
#include "models/llama7b/llama7blayer_operation.h"
#include "models/llama7b/llama7blayer_encoder_operation.h"
#include "models/llama7b/llama7blayer_fusion_operation.h"
#include "models/llama13b/llama13blayer_fusion_operation.h"
#include "models/baichuan1_7b/baichuan1_7b_layer_decoder_operation.h"
#include "models/baichuan1_7b/baichuan1_7b_layer_encoder_operation.h"
#include "models/baichuan1_7b/baichuan1_7b_layer_encoder_with_bias_operation.h"
#include "models/baichuan2_7b/baichuan2_7b_layer_decoder_operation.h"
#include "models/baichuan2_7b/baichuan2_7b_layer_encoder_operation.h"
#include "models/baichuan13b/baichuan13b_layer_decoder_operation.h"
#include "models/baichuan2_13b/baichuan2_13b_layer_decoder_operation.h"
#include "models/baichuan2_13b/baichuan2_13b_layer_encoder_operation.h"
#include "models/baichuan2_7b/baichuan2_7b_layer_decoder_parallel_operation.h"
#include "models/baichuan2_7b/baichuan2_7b_layer_encoder_parallel_operation.h"
#include "models/chatglm2_6b/chatglm2_6b_layer_decoder_operation.h"
#include "models/chatglm2_6b/chatglm2_6b_layer_encoder_operation.h"
#include "models/llama13b/llama13blayer_parallel_operation.h"
#include "models/chatglm2_6b/chatglm2_6b_fusion_layer_decoder_operation.h"
#include "models/chatglm2_6b/chatglm2_6b_fusion_layer_encoder_operation.h"
#include "models/chatglm2_6b/chatglm2_6b_fusion_layer_decoder_parallel_operation.h"
#include "models/bloom7b/bloom7blayer_decoder_operation.h"
#include "models/bloom7b/bloom7blayer_encoder_operation.h"
#include "models/bloom7b/bloom7blayer_parallel_decoder_operation.h"
#include "models/bloom7b/bloom7blayer_parallel_encoder_operation.h"
#include "models/chatglm2_6b/chatglm2_6blayer_decoder_flashattention_operation.h"
#include "models/gptneox20b/gptneox20blayer_embedding_operation.h"
#include "models/gptneox20b/gptneox20blayer_encoder_operation.h"
#include "models/gptneox20b/gptneox20blayer_decoder_operation.h"
#include "models/gptneox20b/gptneox20blayer_decoder_flashattention_operation.h"
#include "models/llama13b/llama13blayer_fusion_quant_operation.h"
#include "models/llama65b/llama65blayer_encoder_parallel_operation.h"
#include "models/llama_adapter7b/llama_adapter_7b_layer_decoder_operation.h"
#include "models/llama_adapter7b/llama_adapter_7b_layer_decoder_adapter_operation.h"
#include "models/llama_adapter7b/llama_adapter_7b_layer_encoder_operation.h"
#include "models/llama_adapter7b/llama_adapter_7b_layer_encoder_adapter_operation.h"

using OperationCreateFunc = std::function<AclTransformer::Operation *(const nlohmann::json &paramJson)>;

static AclTransformer::Operation *LLaMAAdapter7BLayerOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMAAdapter7BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ASD_LOG(INFO) << "LLaMAAdapter7BLayerOperationCreate LLaMAAdapter7BLayerParam headNum:" 
                  << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model;
    return new AclTransformer::LLaMAAdapter7BLayerOperation(param);
}

static AclTransformer::Operation *LLaMAAdapter7BLayerAdapterOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMAAdapter7BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ASD_LOG(INFO) << "LLaMAAdapter7BLayerAdapterOperationCreate LLaMAAdapter7BLayerParam headNum:" 
                  << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model;
    return new AclTransformer::LLaMAAdapter7BLayerAdapterOperation(param);
}

static AclTransformer::Operation *LLaMAAdapter7BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMAAdapter7BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ASD_LOG(INFO) << "LLaMAAdapter7BLayerEncoderOperationCreate LLaMAAdapter7BLayerParam headNum:" 
                  << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model;
    return new AclTransformer::LLaMAAdapter7BLayerEncoderOperation(param);
}

static AclTransformer::Operation *LLaMAAdapter7BLayerEncoderAdapterOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMAAdapter7BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ASD_LOG(INFO) << "LLaMAAdapter7BLayerEncoderAdapterOperationCreate LLaMAAdapter7BLayerParam headNum:" 
                  << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model;
    return new AclTransformer::LLaMAAdapter7BLayerEncoderAdapterOperation(param);
}

static AclTransformer::Operation *ApplyRotaryEmbOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ApplayRotaryEmbParam param;
    ASD_LOG(INFO) << "ApplayRotaryEmbParam Enter";
    return new AclTransformer::ApplyRotaryEmbOperation(param);
}

static AclTransformer::Operation *LLaMA13BLayerFusionQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMA13BLayerFusionQuantParam param;
    param.model = paramJson["model"].get<std::string>();
    param.inputScale_1 = paramJson["inputScale_1"].get<float>();
    param.inputOffset_1 = paramJson["inputOffset_1"].get<int>();
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.inputScale_2 = paramJson["inputScale_2"].get<float>();
    param.inputOffset_2 = paramJson["inputOffset_2"].get<int>();
    param.scale = paramJson["scale"].get<float>();
    param.inputScale_3 = paramJson["inputScale_3"].get<float>();
    param.inputOffset_3 = paramJson["inputOffset_3"].get<int>();
    param.inputScale_4 = paramJson["inputScale_4"].get<float>();
    param.inputOffset_4 = paramJson["inputOffset_4"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    ASD_LOG(INFO) << "LLaMA13BLayerFusionQuantParam headNum:" << param.headNum << ", dk:" << param.dk
                  << ", model:" << param.model << ", rotaryCoeff:" << param.rotaryCoeff;
    return new AclTransformer::LLaMA13BLayerFusionQuantOperation(param);
}

static AclTransformer::Operation *LLaMA7BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMA7BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    ASD_LOG(INFO) << "LLaMA7BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    return new AclTransformer::LLaMA7BLayerEncoderOperation(param);
}

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

static AclTransformer::Operation *LLaMA65BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMA65BLayerParam param;
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
    ASD_LOG(INFO) << "LLaMA65BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    return new AclTransformer::LLaMA65BLayerEncoderOperation(param);
}

static AclTransformer::Operation *LLaMA7BLayerFusionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMA7BLayerFusionParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    ASD_LOG(INFO) << "LLaMA7BLayerFusionParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rotaryCoeff:" << param.rotaryCoeff;
    return new AclTransformer::LLaMA7BLayerFusionOperation(param);
}

static AclTransformer::Operation *LLaMA13BLayerFusionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LLaMA13BLayerFusionParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    ASD_LOG(INFO) << "LLaMA13BLayerFusionParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rotaryCoeff:" << param.rotaryCoeff;
    return new AclTransformer::LLaMA13BLayerFusionOperation(param);
}

static AclTransformer::Operation *BaiChuan27BLayerDecoderParallelOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan27BLayerParallelParam param;
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
    ASD_LOG(INFO) << "BaiChuan27BLayerParallelParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", rank:" << param.rank << ", rankSize:" << param.rankSize;
    return new AclTransformer::BaiChuan27BLayerDecoderParallelOperation(param);
}

static AclTransformer::Operation *BaiChuan27BLayerEncoderParallelOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan27BLayerParallelParam param;
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
    ASD_LOG(INFO) << "BaiChuan27BLayerParallelParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", rank:" << param.rank << ", rankSize:" << param.rankSize;
    return new AclTransformer::BaiChuan27BLayerEncoderParallelOperation(param);
}

static AclTransformer::Operation *BaiChuan17BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan17BLayerParam param;
    bool bias = static_cast<bool>(paramJson.value("bias", false));
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    ASD_LOG(INFO) << "BaiChuan17BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    if (bias) {
        return new AclTransformer::BaiChuan17BLayerEncoderWithBiasOperation(param);
    } else {
        return new AclTransformer::BaiChuan17BLayerEncoderOperation(param);
    }
}

static AclTransformer::Operation *BaiChuan17BLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan17BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ASD_LOG(INFO) << "BaiChuan17BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    return new AclTransformer::BaiChuan17BLayerDecoderOperation(param);
}

static AclTransformer::Operation *BaiChuan27BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan27BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    if (paramJson.contains("transposedWeight")) {
        param.transposedWeight = paramJson["transposedWeight"].get<bool>();
    }
    ASD_LOG(INFO) << "BaiChuan17BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", transposedWeight:" << param.transposedWeight;
    return new AclTransformer::BaiChuan27BLayerEncoderOperation(param);
}

static AclTransformer::Operation *BaiChuan27BLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan27BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    if (paramJson.contains("transposedWeight")) {
        param.transposedWeight = paramJson["transposedWeight"].get<bool>();
    }
    ASD_LOG(INFO) << "BaiChuan17BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", transposedWeight:" << param.transposedWeight;
    return new AclTransformer::BaiChuan27BLayerDecoderOperation(param);
}

static AclTransformer::Operation *BaiChuan13BLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan13BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ASD_LOG(INFO) << "BaiChuan13BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    return new AclTransformer::BaiChuan13BLayerDecoderOperation(param);
}

static AclTransformer::Operation *BaiChuan213BLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan213BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ASD_LOG(INFO) << "BaiChuan213BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    return new AclTransformer::BaiChuan213BLayerDecoderOperation(param);
}

static AclTransformer::Operation *BaiChuan213BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BaiChuan213BLayerParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ASD_LOG(INFO) << "BaiChuan213BLayerParam headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk;
    return new AclTransformer::BaiChuan213BLayerEncoderOperation(param);
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
    if (paramJson.find("allReduceType") != paramJson.end()) {
        param.allReduceType = paramJson["allReduceType"].get<std::string>();
    }
    ASD_LOG(INFO) << "AllReduceParam rank:" << param.rank;
    ASD_LOG(INFO) << "AllReduceParam rankSize:" << param.rankSize;
    return new AclTransformer::AllReduceOperation(param);
}

static AclTransformer::Operation *AllGatherOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AllGatherParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    ASD_LOG(INFO) << "AllGatherParam rank:" << param.rank;
    ASD_LOG(INFO) << "AllGatherParam rankSize:" << param.rankSize;
    ASD_LOG(INFO) << "AllGatherParam backend:" << param.backend;
    return new AclTransformer::AllGatherOperation(param);
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
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<std::int64_t>();
        ASD_LOG(INFO) << "param.headNum: " << param.headNum;
    }
    return new AclTransformer::RopeOperation(param);
}

AclTransformer::Operation *PositionEmbedding1dSplitFusionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::PositionEmbedding1dFusionParam param;
    param.headNum = paramJson["headNum"].get<int64_t>();
    ASD_LOG(INFO) << "param.headNum: " << param.headNum;
    return new AclTransformer::PositionEmbedding1dSplitFusionOperation(param);
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
    if (paramJson.contains("activationFuncType")) {
        param.activationFuncType =
            AclTransformer::FfnParam::ActivationFuncType(paramJson["activationFuncType"].get<int32_t>());
    }
    ASD_LOG(INFO) << "FfnParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias << ", activationFuncType:" << param.activationFuncType;
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
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    ASD_LOG(INFO) << "MlpParam transposeB:" << param.transposeB;
    return new AclTransformer::MlpOperation(param);
}

static AclTransformer::Operation *MlpQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::MlpQuantParam param;
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
        ASD_LOG(INFO) << "MlpQuantParam model:" << param.model;
    } else {
        param.model = "llama7b";
        ASD_LOG(INFO) << "MlpQuantParam is empty, default model:" << param.model;
    }

    param.inputScale = paramJson["inputScale"].get<double>();
    param.inputOffset = paramJson["inputOffset"].get<int>();
    ASD_LOG(INFO) << "MlpQuantParams: "
                  << ", input_scale:" << param.inputScale << ", input_offset:" << param.inputOffset;
    return new AclTransformer::MlpQuantOperation(param);
}

static AclTransformer::Operation *AnyOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AnyParam param;
    param.kernelGraph = paramJson;
    return new AclTransformer::AnyOperation(param);
}

static AclTransformer::Operation *SelfAttentionCrossOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::SelfAttentionCrossParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    ASD_LOG(INFO) << "SelfAttentionCrossParam "
                  << "headNum:" << param.headNum << ", dk:" << param.dk << ", model:" << param.model;
    return new AclTransformer::SelfAttentionCrossOperation(param);
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
    if (paramJson.contains("preScale")) {
        param.preScale = paramJson["preScale"].get<float>();
    }
    if (paramJson.contains("postScale")) {
        param.postScale = paramJson["postScale"].get<float>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("invNormFactorvarAttr")) {
        param.invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();
    }
    ASD_LOG(INFO) << "SelfAttentionKvCacheParam transKey:" << param.transKey << ", headNum:" << param.headNum
                  << ", layerId:" << param.layerId << ", dk:" << param.dk << ", preScale" << param.preScale
                  << ", postScale" << param.postScale << ", model" << param.model << ", hiddenSizePerHead"
                  << param.hiddenSizePerHead;
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
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int64_t>();
        ASD_LOG(INFO) << "PositionEmbeddingParam headNum:" << param.headNum;
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    } else {
        param.model = "llama7b";
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("rotaryPct")) {
        param.rotaryPct = paramJson["rotaryPct"].get<float>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int64_t>();
    }
    if (paramJson.contains("is2d")) {
        param.is2d = paramJson["is2d"].get<bool>();
    }
    if (paramJson.contains("isFusion")) {
        param.isFusion = paramJson["isFusion"].get<bool>();
    }
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
    if (paramJson.contains("preScale")) {
        param.preScale = paramJson["preScale"].get<float>();
    }
    if (paramJson.contains("postScale")) {
        param.postScale = paramJson["postScale"].get<float>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("invNormFactorvarAttr")) {
        param.invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();
    }
    ASD_LOG(INFO) << "SelfAttentionKvCacheParam transKey:" << param.transKey << ", headNum:" << param.headNum
                  << ", layerId:" << param.layerId << ", dk:" << param.dk << ", preScale" << param.preScale
                  << ", postScale" << param.postScale << ", model" << param.model << ", hiddenSizePerHead"
                  << param.hiddenSizePerHead << ", numHeadsPerPartition" << param.numHeadsPerPartition
                  << ", numGroupsPerPartition" << param.numGroupsPerPartition << ", invNormFactorvarAttr"
                  << param.invNormFactorvarAttr;
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

static AclTransformer::Operation *ChatGlm2LayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm2LayerParam param;
    param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    param.layerId = paramJson["layerId"].get<int64_t>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.preScale = paramJson["preScale"].get<float>();
    param.postScale = paramJson["postScale"].get<float>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.model = paramJson["model"].get<std::string>();
    ASD_LOG(INFO) << "ChatGlm2LayerEncoderOperationCreate numHeadsPerPartition:" << param.numHeadsPerPartition
                  << ", numGroupsPerPartition:" << param.numGroupsPerPartition
                  << ", hiddenSizePerHead:" << param.hiddenSizePerHead << ", rmsNormEps:" << param.rmsNormEps
                  << ", layerId:" << param.layerId << ", residualAddScale:" << param.residualAddScale
                  << ", preScale:" << param.preScale << ", postScale:" << param.postScale
                  << ", transKey:" << param.transKey << ", model:" << param.model;
    return new AclTransformer::ChatGlm2LayerEncoderOperation(param);
}

static AclTransformer::Operation *ChatGlm2LayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm2LayerParam param;
    param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    param.layerId = paramJson["layerId"].get<int64_t>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.preScale = paramJson["preScale"].get<float>();
    param.postScale = paramJson["postScale"].get<float>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.model = paramJson["model"].get<std::string>();
    ASD_LOG(INFO) << "ChatGlm2LayerEncoderOperationCreate numHeadsPerPartition:" << param.numHeadsPerPartition
                  << ", numGroupsPerPartition:" << param.numGroupsPerPartition
                  << ", hiddenSizePerHead:" << param.hiddenSizePerHead << ", rmsNormEps:" << param.rmsNormEps
                  << ", layerId:" << param.layerId << ", residualAddScale:" << param.residualAddScale
                  << ", preScale:" << param.preScale << ", postScale:" << param.postScale
                  << ", transKey:" << param.transKey << ", model:" << param.model;
    return new AclTransformer::ChatGlm2LayerDecoderOperation(param);
}

static AclTransformer::Operation *ChatGlm2FusionLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm2LayerParam param;
    param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    param.layerId = paramJson["layerId"].get<int64_t>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.preScale = paramJson["preScale"].get<float>();
    param.postScale = paramJson["postScale"].get<float>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.model = paramJson["model"].get<std::string>();
    ASD_LOG(INFO) << "ChatGlm2LayerEncoderOperationCreate numHeadsPerPartition:" << param.numHeadsPerPartition
                  << ", numGroupsPerPartition:" << param.numGroupsPerPartition
                  << ", hiddenSizePerHead:" << param.hiddenSizePerHead << ", rmsNormEps:" << param.rmsNormEps
                  << ", layerId:" << param.layerId << ", residualAddScale:" << param.residualAddScale
                  << ", preScale:" << param.preScale << ", postScale:" << param.postScale
                  << ", transKey:" << param.transKey << ", model:" << param.model;
    return new AclTransformer::ChatGlm2FusionLayerEncoderOperation(param);
}

static AclTransformer::Operation *ChatGlm2FusionLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm2LayerParam param;
    param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    param.layerId = paramJson["layerId"].get<int64_t>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.preScale = paramJson["preScale"].get<float>();
    param.postScale = paramJson["postScale"].get<float>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.model = paramJson["model"].get<std::string>();
    ASD_LOG(INFO) << "ChatGlm2LayerEncoderOperationCreate numHeadsPerPartition:" << param.numHeadsPerPartition
                  << ", numGroupsPerPartition:" << param.numGroupsPerPartition
                  << ", hiddenSizePerHead:" << param.hiddenSizePerHead << ", rmsNormEps:" << param.rmsNormEps
                  << ", layerId:" << param.layerId << ", residualAddScale:" << param.residualAddScale
                  << ", preScale:" << param.preScale << ", postScale:" << param.postScale
                  << ", transKey:" << param.transKey << ", model:" << param.model;
    return new AclTransformer::ChatGlm2FusionLayerDecoderOperation(param);
}

static AclTransformer::Operation *ChatGlm2FusionLayerDecoderParallelOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm2LayerParam param;
    param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    param.layerId = paramJson["layerId"].get<int64_t>();
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.preScale = paramJson["preScale"].get<float>();
    param.postScale = paramJson["postScale"].get<float>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.model = paramJson["model"].get<std::string>();
    param.rank = paramJson["rank"].get<int64_t>();
    param.rankSize = paramJson["rankSize"].get<int64_t>();
    ASD_LOG(INFO) << "ChatGlm2LayerEncoderOperationCreate numHeadsPerPartition:" << param.numHeadsPerPartition
                  << ", numGroupsPerPartition:" << param.numGroupsPerPartition
                  << ", hiddenSizePerHead:" << param.hiddenSizePerHead << ", rmsNormEps:" << param.rmsNormEps
                  << ", layerId:" << param.layerId << ", residualAddScale:" << param.residualAddScale
                  << ", preScale:" << param.preScale << ", postScale:" << param.postScale
                  << ", transKey:" << param.transKey << ", model:" << param.model << ", rank:" << param.rank
                  << ", rankSize:" << param.rankSize;
    return new AclTransformer::ChatGlm2FusionLayerDecoderParallelOperation(param);
}

static AclTransformer::Operation *ChatGlm2LayerDecoderFlashAttentionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm2LayerDecoderFlashAttentionParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.is2d = paramJson["is2d"].get<bool>();
    param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int32_t>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.model = paramJson["model"].get<std::string>();
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int32_t>());
    }

    ASD_LOG(INFO) << "ChatGlm2LayerDecoderFlashAttentionOperation" << param.model;
    return new AclTransformer::ChatGlm2LayerDecoderFlashAttentionOperation(param);
}

static AclTransformer::Operation *Bloom7BLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::Bloom7BLayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();
    param.activationFuncType = paramJson["activationFuncType"].get<int>();
    ASD_LOG(INFO) << "Bloom7BLayerDecoderOperationCreate layerNormEps:" << param.layerNormEps
                  << ", headNum:" << param.headNum << ", dk:" << param.dk
                  << ", invNormFactorvarAttr:" << param.invNormFactorvarAttr
                  << ", activationFuncType:" << param.activationFuncType;
    return new AclTransformer::Bloom7BLayerDecoderOperation(param);
}

static AclTransformer::Operation *Bloom7BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::Bloom7BLayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();
    param.activationFuncType = paramJson["activationFuncType"].get<int>();
    ASD_LOG(INFO) << "Bloom7BLayerEncoderOperationCreate layerNormEps:" << param.layerNormEps
                  << ", headNum:" << param.headNum << ", dk:" << param.dk
                  << ", invNormFactorvarAttr:" << param.invNormFactorvarAttr
                  << ", activationFuncType:" << param.activationFuncType;
    return new AclTransformer::Bloom7BLayerEncoderOperation(param);
}

static AclTransformer::Operation *Bloom7BLayerParallelDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::Bloom7BLayerParallelParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();
    param.activationFuncType = paramJson["activationFuncType"].get<int>();
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    ASD_LOG(INFO) << "Bloom7BLayerParallelDecoderOperationCreate layerNormEps:" << param.layerNormEps
                  << ", headNum:" << param.headNum << ", dk:" << param.dk
                  << ", invNormFactorvarAttr:" << param.invNormFactorvarAttr
                  << ", activationFuncType:" << param.activationFuncType
                  << ", rank" << param.rank << ", rankSize" << param.rankSize;
    return new AclTransformer::Bloom7BLayerParallelDecoderOperation(param);
}

static AclTransformer::Operation *Bloom7BLayerParallelEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::Bloom7BLayerParallelParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();
    param.activationFuncType = paramJson["activationFuncType"].get<int>();
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    ASD_LOG(INFO) << "Bloom7BLayerParallelEncoderOperationCreate layerNormEps:" << param.layerNormEps
                  << ", headNum:" << param.headNum << ", dk:" << param.dk
                  << ", invNormFactorvarAttr:" << param.invNormFactorvarAttr
                  << ", activationFuncType:" << param.activationFuncType
                  << ", rank" << param.rank << ", rankSize" << param.rankSize;
    return new AclTransformer::Bloom7BLayerParallelEncoderOperation(param);
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

static AclTransformer::Operation *TransdataInt8OperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::TransDataInt8Param param;
    return new AclTransformer::TransDataInt8Operation(param);
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
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
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

AclTransformer::Operation *Glm130BLayerDecoderFusionOperationCreate(const nlohmann::json &paramJson)
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
    return new AclTransformer::Glm130BLayerDecoderFusionOperation(param);
}

AclTransformer::Operation *WordEmbeddingParallelOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::WordEmbeddingParallelParam param;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("rankRoot")) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    return new AclTransformer::WordEmbeddingParallelOperation(param);
}

static AclTransformer::Operation *GptNeox20BLayerEmbeddingOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::GptNeox20BLayerEmbeddingParam param;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int>();
    }
    ASD_LOG(INFO) << "GptNeox20BLayerEmbeddingParam axis:" << param.axis;
    return new AclTransformer::GptNeox20BLayerEmbeddingOperation(param);
}

static AclTransformer::Operation *GptNeox20BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::GptNeox20BLayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.rotaryPct = paramJson["rotaryPct"].get<float>();
    ASD_LOG(INFO) << "GptNeox20BLayerParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", rotaryPct:" << param.rotaryPct;
    return new AclTransformer::GptNeox20BLayerEncoderOperation(param);
}

static AclTransformer::Operation *GptNeox20BLayerDecoderOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::GptNeox20BLayerParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.rotaryPct = paramJson["rotaryPct"].get<float>();
    ASD_LOG(INFO) << "GptNeox20BLayerParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", rotaryPct:" << param.rotaryPct;
    return new AclTransformer::GptNeox20BLayerDecoderOperation(param);
}

static AclTransformer::Operation *GptNeox20BLayerDecoderFlashAttentionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::GptNeox20BLayerDecoderFlashAttentionParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.rotaryPct = paramJson["rotaryPct"].get<float>();
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    ASD_LOG(INFO) << "GptNeox20BLayerDecoderFlashAttentionParam layerNormEps:" << param.layerNormEps
                  << ", headNum:" << param.headNum << ", transKey:" << param.transKey << ", dk:" << param.dk
                  << ", layerId:" << param.layerId << ", rotaryPct:" << param.rotaryPct;
    return new AclTransformer::GptNeox20BLayerDecoderFlashAttentionOperation(param);
}

AclTransformer::Operation *LmHeadOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LmHeadParam param;
    return new AclTransformer::LmHeadOperation(param);
}

AclTransformer::Operation *LmHeadSliceOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LmHeadSliceParam param;
    if (paramJson.contains("seqLen")) {
        param.seqLen = paramJson["seqLen"].get<int>();
    }
    return new AclTransformer::LmHeadSliceOperation(param);
}

AclTransformer::Operation *LmHeadParallelOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LmHeadParallelParam param;
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("rankRoot")) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    return new AclTransformer::LmHeadParallelOperation(param);
}

static AclTransformer::Operation *RmsPreNormQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::RmsPreNormQuantParam param;
    param.inputScale = paramJson["inputScale"].get<double>();
    param.inputOffset = paramJson["inputOffset"].get<int>();
    return new AclTransformer::RmsPreNormQuantOperation(param);
}

static AclTransformer::Operation *RmsNormQuantOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::RmsNormQuantParam param;
    param.inputScale = paramJson["inputScale"].get<double>();
    param.inputOffset = paramJson["inputOffset"].get<int>();
    return new AclTransformer::RmsNormQuantOperation(param);
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"PostOperation", &PostOperationCreate},
    {"RmsPreNormQuantOperation", &RmsPreNormQuantOperationCreate},
    {"RmsNormQuantOperation", &RmsNormQuantOperationCreate},
    {"AllReduceOperation", &AllReduceOperationCreate},
    {"AllGatherOperation", &AllGatherOperationCreate},
    {"LinearParallelOperation", &LinearParallelOperationCreate},
    {"AddOperation", &AddOperationCreate},
    {"NormOperation", &NormOperationCreate},
    {"RopeOperation", &RopeOperationCreate},
    {"PositionEmbedding1dSplitFusionOperation", &PositionEmbedding1dSplitFusionOperationCreate},
    {"AddNormOperation", &AddNormOperationCreate},
    {"RmsNormOperation", &RmsNormOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"ApplyRotaryEmbOperation", &ApplyRotaryEmbOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"MatmulOperation", &MatmulOperationCreate},
    {"FfnOperation", &FfnOperationCreate},
    {"MlpOperation", &MlpOperationCreate},
    {"MlpQuantOperation", &MlpQuantOperationCreate},
    {"EmbeddingOperation", &EmbeddingOperationCreate},
    {"PositionEmbedding1dSplitOperation", &PositionEmbedding1dSplitOperationCreate},
    {"PositionEmbeddingOperation", &PositionEmbeddingOperationCreate},
    {"SelfAttentionKvCacheOperation", &SelfAttentionKvCacheOperationCreate},
    {"SelfAttentionKvCacheFusionOperation", &SelfAttentionKvCacheFusionOperationCreate},
    {"SelfAttentionOperation", &SelfAttentionOperationCreate},
    {"SelfAttentionCrossOperation", &SelfAttentionCrossOperationCreate},
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
    {"TransDataInt8Operation", &TransdataInt8OperationCreate},
    {"ChatGlm6BLayerDecoderQuantOperation", &ChatGlm6BLayerDecoderQuantOperationCreate},
    {"ChatGlm6BLayerDecoderLastQuantOperation", &ChatGlm6BLayerDecoderLastQuantOperationCreate},
    {"ChatGlm6BLayerDecoderFirstQuantOperation", &ChatGlm6BLayerDecoderFirstQuantOperationCreate},
    {"ChatGlm6BLayerDecoderFlashAttentionOperation", &ChatGlm6BLayeEncoderFlashAttentionOperationCreate},
    {"Glm130BLayerDecoderFusionOperation", &Glm130BLayerDecoderFusionOperationCreate},
    {"Glm130BLayerDecoderOperation", &Glm130BLayerDecoderOperationCreate},
    {"LLaMA7BLayerOperation", &LLaMA7BLayerOperationCreate},
    {"LLaMA7BLayerEncoderOperation", &LLaMA7BLayerEncoderOperationCreate},
    {"LLaMA7BLayerFusionOperation", &LLaMA7BLayerFusionOperationCreate},
    {"LLaMA13BLayerFusionOperation", &LLaMA13BLayerFusionOperationCreate},
    {"BaiChuan17BLayerDecoderOperation", &BaiChuan17BLayerDecoderOperationCreate},
    {"BaiChuan17BLayerEncoderOperation", &BaiChuan17BLayerEncoderOperationCreate},
    {"BaiChuan27BLayerDecoderOperation", &BaiChuan27BLayerDecoderOperationCreate},
    {"BaiChuan27BLayerEncoderOperation", &BaiChuan27BLayerEncoderOperationCreate},
    {"BaiChuan213BLayerEncoderOperation", &BaiChuan213BLayerEncoderOperationCreate},
    {"BaiChuan13BLayerDecoderOperation", &BaiChuan13BLayerDecoderOperationCreate},
    {"BaiChuan213BLayerDecoderOperation", &BaiChuan213BLayerDecoderOperationCreate},
    {"BaiChuan27BLayerDecoderParallelOperation", &BaiChuan27BLayerDecoderParallelOperationCreate},
    {"BaiChuan27BLayerEncoderParallelOperation", &BaiChuan27BLayerEncoderParallelOperationCreate},
    {"LmHeadOperation", &LmHeadOperationCreate},
    {"LmHeadSliceOperation", &LmHeadSliceOperationCreate},
    {"LmHeadParallelOperation", &LmHeadParallelOperationCreate},
    {"ChatGlm2LayerEncoderOperation", &ChatGlm2LayerEncoderOperationCreate},
    {"ChatGlm2LayerDecoderOperation", &ChatGlm2LayerDecoderOperationCreate},
    {"LLaMA13BLayerOperation", &LLaMA13BLayerOperationCreate},
    {"LLaMA65BLayerEncoderOperation", &LLaMA65BLayerEncoderOperationCreate},
    {"ChatGlm2FusionLayerEncoderOperation", &ChatGlm2FusionLayerEncoderOperationCreate},
    {"ChatGlm2FusionLayerDecoderOperation", &ChatGlm2FusionLayerDecoderOperationCreate},
    {"ChatGlm2FusionLayerDecoderParallelOperation", &ChatGlm2FusionLayerDecoderParallelOperationCreate},
    {"Bloom7BLayerDecoderOperation", &Bloom7BLayerDecoderOperationCreate},
    {"Bloom7BLayerEncoderOperation", &Bloom7BLayerEncoderOperationCreate},
    {"Bloom7BLayerParallelDecoderOperation", &Bloom7BLayerParallelDecoderOperationCreate},
    {"Bloom7BLayerParallelEncoderOperation", &Bloom7BLayerParallelEncoderOperationCreate},
    {"ChatGlm2LayerDecoderFlashAttentionOperation", &ChatGlm2LayerDecoderFlashAttentionOperationCreate},
    {"WordEmbeddingParallelOperation", &WordEmbeddingParallelOperationCreate},
    {"GptNeox20BLayerEmbeddingOperation", &GptNeox20BLayerEmbeddingOperationCreate},
    {"GptNeox20BLayerEncoderOperation", &GptNeox20BLayerEncoderOperationCreate},
    {"GptNeox20BLayerDecoderOperation", &GptNeox20BLayerDecoderOperationCreate},
    {"GptNeox20BLayerDecoderFlashAttentionOperation", &GptNeox20BLayerDecoderFlashAttentionOperationCreate},
    {"LLaMA13BLayerFusionQuantOperation", &LLaMA13BLayerFusionQuantOperationCreate},
    {"LLaMAAdapter7BLayerOperation", &LLaMAAdapter7BLayerOperationCreate},
    {"LLaMAAdapter7BLayerAdapterOperation", &LLaMAAdapter7BLayerAdapterOperationCreate},
    {"LLaMAAdapter7BLayerEncoderOperation", &LLaMAAdapter7BLayerEncoderOperationCreate},
    {"LLaMAAdapter7BLayerEncoderAdapterOperation", &LLaMAAdapter7BLayerEncoderAdapterOperationCreate}};

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

    if (opName == "ChatGlm6BLayerDecoderFlashAttentionOperation" || opName == "LLaMA7BLayerFusionOperation" ||
        opName == "ChatGlm2LayerDecoderFlashAttentionOperation" || opName == "Glm130BLayerDecoderFusionOperation" ||
        opName == "GptNeox20BLayerDecoderFlashAttentionOperation" || opName == "LLaMA13BLayerFusionOperation") {
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