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
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/any_operation.h"
#include "examples/ops/chatglm6b/chatglm6b_operation.h"
#include "examples/ops/chatglm6b/chatglm6b28_operation.h"
#include "examples/ops/chatglm6b/chatglm6b_fusion_operation.h"
#include "examples/ops/bert/bert_operation.h"
#include "examples/ops/llama7b/llama7b_operation.h"

using OperationCreateFunc = std::function<AclTransformer::Operation *(const nlohmann::json &paramJson)>;

AclTransformer::Operation *AddOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AddParam param;
    if (paramJson.find("scale") != paramJson.end()) {
        param.scale = paramJson["scale"].get<float>();
    }
    ASD_LOG(INFO) << "AddParam scale:" << param.scale;
    return new AclTransformer::AddOperation(param);
}

AclTransformer::Operation *AddNormOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AddNormParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    ASD_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps;
    return new AclTransformer::AddNormOperation(param);
}

AclTransformer::Operation *RmsNormOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::RmsNormParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<double>();
    return new AclTransformer::RmsNormOperation(param);
}

AclTransformer::Operation *NormOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::NormParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    ASD_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps;
    return new AclTransformer::NormOperation(param);
}

AclTransformer::Operation *LinearOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::LinearParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    ASD_LOG(INFO) << "LinearParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB;
    return new AclTransformer::LinearOperation(param);
}

AclTransformer::Operation *FfnOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::FfnParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    ASD_LOG(INFO) << "FfnParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB;
    return new AclTransformer::FfnOperation(param);
}

AclTransformer::Operation *MlpOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::MlpParam param;
    return new AclTransformer::MlpOperation(param);
}

AclTransformer::Operation *AnyOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::AnyParam param;
    param.kernelGraph = paramJson;
    return new AclTransformer::AnyOperation(param);
}

AclTransformer::Operation *SelfAttentionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::SelfAttentionParam param;
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.headNum = paramJson["headNum"].get<int>();
    ASD_LOG(INFO) << "PositionEmbeddingParam transKey:" << param.transKey << ", dk:" << param.dk
                  << ", headNum:" << param.headNum;
    return new AclTransformer::SelfAttentionOperation(param);
}

AclTransformer::Operation *PositionEmbedding1dSplitOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::PositionEmbedding1dSplitParam param;
    param.headNum = paramJson["headNum"].get<int>();
    ASD_LOG(INFO) << "PositionEmbeddingParam headNum:" << param.headNum;
    return new AclTransformer::PositionEmbedding1dSplitOperation(param);
}

AclTransformer::Operation *PositionEmbeddingOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::PositionEmbeddingParam param;
    param.headNum = paramJson["headNum"].get<int>();
    ASD_LOG(INFO) << "PositionEmbeddingParam headNum:" << param.headNum;
    return new AclTransformer::PositionEmbeddingOperation(param);
}

AclTransformer::Operation *SelfAttentionKvCacheOperationCreate(const nlohmann::json &paramJson)
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

AclTransformer::Operation *TransposeOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::TransposeParam param;
    param.dimA = paramJson["dimA"].get<int>();
    param.dimB = paramJson["dimB"].get<int>();
    ASD_LOG(INFO) << "transpose(" << param.dimA << "," << param.dimB << ")";
    return new AclTransformer::TransposeOperation(param);
}

AclTransformer::Operation *BertOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::BertParam param;
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.headNum = paramJson["headNum"].get<int>();
    ASD_LOG(INFO) << "BertParam transKey:" << param.transKey << ", dk" << param.dk << ", headNum:" << param.headNum;
    return new AclTransformer::BertOperation(param);
}

AclTransformer::Operation *ChatGlm6BOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    ASD_LOG(INFO) << "ChatGlm6BParam layerNormEps:" << param.layerNormEps << ", headNum" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale;
    return new AclTransformer::ChatGlm6BOperation(param);
}

AclTransformer::Operation *ChatGlm6B28OperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6B28Param param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerCount = paramJson["layerCount"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    ASD_LOG(INFO) << "ChatGlm6BParam layerNormEps:" << param.layerNormEps << ", headNum" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerCount:" << param.layerCount
                  << ", residualAddScale:" << param.residualAddScale;
    return new AclTransformer::ChatGlm6B28Operation(param);
}

AclTransformer::Operation *ChatGlm6BFusionOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::ChatGlm6BFusionParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    ASD_LOG(INFO) << "ChatGlm6BFusionParam layerNormEps:" << param.layerNormEps << ", headNum" << param.headNum
                  << ", residualAddScale:" << param.residualAddScale;
    return new AclTransformer::ChatGlm6BFusionOperation(param);
}

AclTransformer::Operation *Llam7BOperationCreate(const nlohmann::json &paramJson)
{
    AclTransformer::Llam7BParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.model = paramJson["model"].get<std::string>();

    ASD_LOG(INFO) << "Llam7BParam rmsNormEps:" << param.rmsNormEps << ", dk" << param.dk
                  << ", headNum:" << param.headNum << ", model:" << param.model;
    return new AclTransformer::Llam7BOperation(param);
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"AddOperation", &AddOperationCreate},
    {"NormOperation", &NormOperationCreate},
    {"AddNormOperation", &AddNormOperationCreate},
    {"RmsNormOperation", &RmsNormOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"FfnOperation", &FfnOperationCreate},
    {"MlpOperation", &MlpOperationCreate},
    {"PositionEmbedding1dSplitOperation", &PositionEmbedding1dSplitOperationCreate},
    {"PositionEmbeddingOperation", &PositionEmbeddingOperationCreate},
    {"SelfAttentionKvCacheOperation", &SelfAttentionKvCacheOperationCreate},
    {"SelfAttentionOperation", &SelfAttentionOperationCreate},
    {"AnyOperation", &AnyOperationCreate},
    {"BertOperation", &BertOperationCreate},
    {"ChatGlm6BOperation", &ChatGlm6BOperationCreate},
    {"ChatGlm6B28Operation", &ChatGlm6B28OperationCreate},
    {"ChatGlm6BFusionOperation", &ChatGlm6BFusionOperationCreate},
    {"Llam7BOperation", &Llam7BOperationCreate},
};

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