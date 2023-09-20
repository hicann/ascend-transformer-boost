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
#include <set>

namespace AclTransformer {
class Config {
public:
    Config();
    ~Config();
    static std::string GetSaveTensorDir();
    bool IsSaveTensor();
    bool IsSaveTensorByRange();
    void DisableSaveTensor();
    void EnableSaveTensor();
    uint64_t GetSaveTensorMaxNum();
    uint64_t GetSaveTensorMinNum();
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
    bool IsSelfAttentionCrossOpsRunnerEnable();
    bool IsPositionEmbedding1dSplitOpsRunnerEnable();
    bool IsTransposeOpsRunnerEnable();
    bool IsStreamSyncEveryKernelEnable();
    bool IsStreamSyncEveryRunnerEnable();
    bool IsStreamSyncEveryPlanEnable();
    bool IsSkipKernel(const std::string &kernelName);
    bool Is910B();
    bool IsOpsRunnerSetupCacheEnable();
    bool IsOpsRunnerKernelCacheEnable();
    bool IsConvertNCHWToND() const;
    bool IsSaveTensorForRunner(const std::string &runnerName);
    bool IsTorchTensorFormatCast();
    bool IsUsingProfiling();
    bool IsTilingCopyStreamEnable();

private:
    static bool IsEnable(const char *env, bool enable = false);
    void InitSkipKernelName();
    void InitWorkspaceSize();
    void InitIs910B();
    void InitSaveTensor();
    void InitSaveTensor(const char *env, std::set<std::string> &nameSet);

private:
    bool isSaveTensor_ = false;
    bool isSaveTensorByRange_ = false;
    uint64_t saveTensorMaxNum_ = 1;
    uint64_t saveTensorMinNum_ = 1;
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
    bool isSelfAttentionCrossOpsRunnerEnable_ = false;
    bool isPositionEmbedding1dSplitOpsRunnerEnable_ = false;
    bool isTransposeOpsRunnerEnable_ = false;
    bool isStreamSyncEveryKernelEnable_ = false;
    bool isStreamSyncEveryRunnerEnable_ = false;
    bool isStreamSyncEveryPlanEnable_ = false;
    std::vector<std::string> skipKernelNames_;
    bool is910B_ = false;
    bool isOpsRunnerSetupCacheEnable_ = false;
    bool isOpsRunnerKernelCacheEnable_ = false;
    bool isUsePpMatmul_ = false;
    bool isConvertNCHWToND_ = false;
    bool isTorchTensorFormatCast_ = true;
    bool isUsingProfiling_ = false;
    bool isTilingCopyStreamEnable_ = false;
    std::set<std::string> saveTensorRunnerNameSet_;
};
} // namespace AclTransformer
#endif