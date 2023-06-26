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
#include <vector>

namespace AclTransformer {
class Config {
public:
    Config();
    ~Config();
    static std::string GetSaveTensorDir();
    bool IsSaveTensor();
    bool IsAddOpsRunnerEnable();
    bool IsAddNormOpsRunnerEnable();
    bool IsRmsNormOpsRunnerEnable();
    bool IsFfnOpsRunnerEnable();
    bool IsLinearOpsRunnerEnable();
    bool IsNormOpsRunnerEnable();
    bool IsMlpOpsRunnerEnable();
    bool IsPositionEmbeddingOpsRunnerEnable();
    bool IsSelfAttentionKVCacheOpsRunnerEnable();
    bool IsSelfAttentionOpsRunnerEnable();
    bool IsPositionEmbedding1dSplitOpsRunnerEnable();
    bool IsTransposeOpsRunnerEnable();
    bool IsStreamSyncEveryRunnerEnable();
    bool IsStreamSyncEveryKernelEnable();
    bool IsStreamSyncEveryOperationEnable();
    bool IsStreamSyncEveryPlanEnable();
    bool IsKernelCacheEnable();
    bool IsSkipKernel(const std::string &kernelName);
    uint64_t GetWorkspaceSize();
    bool Is910B();
    bool IsOpsRunnerSetupCacheEnable();
    bool IsOpsRunnerWorkspaceReusageEnable();

private:
    static bool IsEnable(const char *env, bool enable = false);
    void InitSkipKernelName();
    void InitWorkspaceSize();
    void InitIs910B();

private:
    bool isSaveTensor_ = false;
    bool isAddOpsRunnerEnable_ = false;
    bool isAddNormOpsRunnerEnable_ = false;
    bool isRmsNormOpsRunnerEnable_ = false;
    bool isFfnOpsRunnerEnable_ = false;
    bool isLinearOpsRunnerEnable_ = false;
    bool isNormOpsRunnerEnable_ = false;
    bool isMlpOpsRunnerEnable_ = false;
    bool isPositionEmbeddingOpsRunnerEnable_ = false;
    bool isSelfAttentionKVCacheOpsRunnerEnable_ = false;
    bool isSelfAttentionOpsRunnerEnable_ = false;
    bool isPositionEmbedding1dSplitOpsRunnerEnable_ = false;
    bool isTransposeOpsRunnerEnable_ = false;
    bool isStreamSyncEveryRunnerEnable_ = false;
    bool isStreamSyncEveryKernelEnable_ = false;
    bool isStreamSyncEveryOperationEnable_ = false;
    bool isStreamSyncEveryPlanEnable_ = false;
    bool isKernelCacheEnable_ = false;
    std::vector<std::string> skipKernelNames_;
    uint64_t workspaceSize_ = 1024 * 1024 * 500;
    bool is910B_ = false;
    bool isOpsRunnerSetupCacheEnable_ = false;
    bool isOpsRunnerWorkspaceReusageEnable_ = true;
};
} // namespace AclTransformer
#endif