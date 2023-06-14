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
#ifndef ACLTRANSFORMER_CONFIG_H
#define ACLTRANSFORMER_CONFIG_H
#include <string>

namespace AclTransformer {
class Config {
public:
    static std::string GetSaveTensorDir();
    static bool IsSaveTensor();
    static bool IsAddOpsRunnerEnable();
    static bool IsAddNormOpsRunnerEnable();
    static bool IsRmsNormOpsRunnerEnable();
    static bool IsFfnOpsRunnerEnable();
    static bool IsLinearOpsRunnerEnable();
    static bool IsNormOpsRunnerEnable();
    static bool IsMlpOpsRunnerEnable();
    static bool IsPositionEmbeddingOpsRunnerEnable();
    static bool IsSelfAttentionKVCacheOpsRunnerEnable();
    static bool IsSelfAttentionOpsRunnerEnable();
    static bool IsPositionEmbedding1dSplitOpsRunnerEnable();
    static bool IsTransposeOpsRunnerEnable();

private:
    static bool IsEnable(const char *env);
};
} // namespace AclTransformer
#endif