/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include <json/json.h>
#include <functional>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/ffn_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"

using OperationCreateFunc = std::function<AclTransformer::Operation *(const Json::Value &paramJson)>;

AclTransformer::Operation *AddOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::AddParam param;
    if (paramJson.isMember("scale")) {
        param.scale = paramJson["scale"].asFloat();
    }
    return new AclTransformer::AddOperation(param);
}

AclTransformer::Operation *AddNormOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::AddNormParam param;
    param.layerNormEps = paramJson["layerNormEps"].asDouble();
    return new AclTransformer::AddNormOperation(param);
}

AclTransformer::Operation *NormOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::NormParam param;
    param.layerNormEps = paramJson["layerNormEps"].asDouble();
    return new AclTransformer::NormOperation(param);
}

AclTransformer::Operation *LinearOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::LinearParam param;
    param.transposeA = paramJson["transposeA"].asBool();
    param.transposeB = paramJson["transposeB"].asBool();
    return new AclTransformer::LinearOperation(param);
}

AclTransformer::Operation *FfnOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::FfnParam param;
    return new AclTransformer::FfnOperation(param);
}

AclTransformer::Operation *SelfAttentionOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::SelfAttentionParam param;
    param.transKey = paramJson["transKey"].asBool();
    param.dk = paramJson["dk"].asInt();
    param.headNum = paramJson["headNum"].asInt();
    return new AclTransformer::SelfAttentionOperation(param);
}

AclTransformer::Operation *PositionEmbeddingOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::PositionEmbeddingParam param;
    param.headNum = paramJson["headNum"].asInt();
    return new AclTransformer::PositionEmbeddingOperation(param);
}

AclTransformer::Operation *SelfAttentionKvCacheOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    param.transKey = paramJson["transKey"].asBool();
    param.headNum = paramJson["headNum"].asInt64();
    param.layerId = paramJson["layerId"].asInt64();
    param.dk = paramJson["dk"].asInt64();
    return new AclTransformer::SelfAttentionKvCacheOperation(param);
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"AddOperation", &AddOperationCreate},
    {"AddNormOperation", &AddNormOperationCreate},
    {"NormOperation", &NormOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"FfnOperation", &FfnOperationCreate},
    {"PositionEmbeddingOperation", &PositionEmbeddingOperationCreate},
    {"SelfAttentionKvCacheOperation", &SelfAttentionKvCacheOperationCreate},
    {"SelfAttentionOperation", &SelfAttentionOperationCreate}};

AclTransformer::Operation *CreateOperation(const std::string &opName, const std::string &param)
{
    Json::Reader reader;
    Json::Value paramJson;
    if (!reader.parse(param, paramJson)) {
        ASD_LOG(ERROR) << "json parse error, CallOp fail";
        return nullptr;
    }

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