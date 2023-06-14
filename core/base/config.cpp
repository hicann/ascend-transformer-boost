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
#include "acltransformer/config.h"
#include <string>

namespace AclTransformer {
std::string Config::GetSaveTensorDir() { return "tensors"; }

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

bool Config::IsFfnOpsRunnerEnable() { return IsEnable("ACLTRANSFORMER_FFN_OPSRUNNER_ENABLE"); }

bool Config::IsLinearOpsRunnerEnable() { return IsEnable("ACLTRANSFORMER_LINEAR_OPSRUNNER_ENABLE"); }

bool Config::IsNormOpsRunnerEnable() { return IsEnable("ACLTRANSFORMER_NORM_OPSRUNNER_ENABLE"); }

bool Config::IsPositionEmbeddingOpsRunnerEnable()
{
    return IsEnable("ACLTRANSFORMER_POSITIONEMBEDDING_OPSRUNNER_ENABLE");
}

bool Config::IsPositionEmbedding1dSplitOpsRunnerEnable()
{
    return IsEnable("ACLTRANSFORMER_POSITIONEMBEDDING_1D_SPLIT_OPSRUNNER_ENABLE");
}

bool Config::IsSelfAttentionKVCacheOpsRunnerEnable()
{
    return IsEnable("ACLTRANSFORMER_SELFATTENTIONKVCACHE_OPSRUNNER_ENABLE");
}

bool Config::IsSelfAttentionOpsRunnerEnable() { return IsEnable("ACLTRANSFORMER_SELFATTENTION_OPSRUNNER_ENABLE"); }

} // namespace AclTransformer