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

using OperationCreateFunc = std::function<AclTransformer::Operation *(const Json::Value &paramJson)>;

AclTransformer::Operation *AddOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::AddParam param;
    param.scale = paramJson["scale"].asFloat();
    return new AclTransformer::AddOperation(param);
}

AclTransformer::Operation *AddNormOperationCreate(const Json::Value &paramJson)
{
    AclTransformer::AddNormParam param;
    param.layerNormEps = paramJson["layerNormEps"].asDouble();
    for (int i = 0; i < paramJson["dims"].size(); i++) {
        param.dims.push_back(paramJson["dims"][i].asInt());
    }
    return new AclTransformer::AddNormOperation(param);
}

std::map<std::string, OperationCreateFunc> g_funcMap = {{"AddOperation", &AddOperationCreate},
                                                        {"AddNormOperation", &AddNormOperationCreate}};

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

    return it->second(paramJson);
}