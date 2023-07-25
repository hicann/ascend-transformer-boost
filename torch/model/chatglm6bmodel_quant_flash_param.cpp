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
#include "chatglm6bmodel_quant_flash_param.h"
#include <nlohmann/json.hpp>

void ChatGlm6BModelQuantFlashParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
    qkvInputScale = paramJson["qkvInputScale"].get<std::vector<float>>();
    qkvInputOffset = paramJson["qkvInputOffset"].get<std::vector<int>>();
    denseInputScale = paramJson["denseInputScale"].get<std::vector<float>>();
    denseInputOffset = paramJson["denseInputOffset"].get<std::vector<int>>();
    selfLnInputScale = paramJson["selfLnInputScale"].get<std::vector<float>>();
    selfLnInputOffset = paramJson["selfLnInputOffset"].get<std::vector<int>>();
    ffnOutInputScale = paramJson["ffnOutInputScale"].get<std::vector<float>>();
    ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<std::vector<int>>();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
}