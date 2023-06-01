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
#include "acltransformer/config.h"
#include <string>

namespace AclTransformer {
bool Config::IsEnable(const char *env)
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return false;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsSaveTensor() { return IsEnable("ACLTRANSFORMER_SAVE_TENSOR"); }

bool Config::IsAddOpsRunnerEnable() { return IsEnable("ACLTRANSFORMER_ADD_OPSRUNNER_ENABLE"); }

bool Config::IsAddNormOpsRunnerEnable() { return IsEnable("ACLTRANSFORMER_ADDNORM_OPSRUNNER_ENABLE"); }

bool Config::IsLinearOpsRunnerEnable() { return IsEnable("ACLTRANSFORMER_LINEAR_OPSRUNNER_ENABLE"); }
} // namespace AclTransformer