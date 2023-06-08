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
#include "layer.h"
#include <functional>
#include <json/json.h>
#include <asdops/utils/log/log.h>

using LayerFunc = std::function<void(const Json::Value &paramJson, AclTransformer::VariantPack &variantPack)>;

void BertLayer(const Json::Value &paramJson, AclTransformer::VariantPack &variantPack);
void BertSelfAttentionLayer(const Json::Value &paramJson, AclTransformer::VariantPack &variantPack);
void BertOutputAttentionLayer(const Json::Value &paramJson, AclTransformer::VariantPack &variantPack);
void ChatGlm6BLayer(const Json::Value &paramJson, AclTransformer::VariantPack &variantPack);

std::map<std::string, LayerFunc> g_layerMap = {
    {"BertLayer", &BertLayer},
    {"BertSelfAttentionLayer", &BertSelfAttentionLayer},
    {"BertOutputAttentionLayer", &BertOutputAttentionLayer},
    {"ChatGlm6BLayer", &ChatGlm6BLayer},
};

bool ExecuteLayer(const std::string &layerName, const std::string &param, AclTransformer::VariantPack &variantPack)
{
    auto it = g_layerMap.find(layerName);
    if (it == g_layerMap.end()) {
        return false;
    }

    Json::Reader reader;
    Json::Value paramJson;
    if (!reader.parse(param, paramJson)) {
        ASD_LOG(ERROR) << " invalid json:" << param;
        return false;
    }

    it->second(paramJson, variantPack);
    return true;
}
