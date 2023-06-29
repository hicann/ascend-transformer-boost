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
#include <iostream>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/strings/match.h>
#include <asdops/utils/strings/str_split.h>

namespace AclTransformer {
Config::Config()
{
    InitSkipKernelName();
    InitWorkspaceSize();
    InitIs910B();
    isSaveTensor_ = IsEnable("ACLTRANSFORMER_SAVE_TENSOR");
    isAddOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_ADD_OPSRUNNER_ENABLE");
    isAddNormOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_ADDNORM_OPSRUNNER_ENABLE");
    isRmsNormOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_RMSNORM_OPSRUNNER_ENABLE");
    isFfnOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_FFN_OPSRUNNER_ENABLE");
    isLinearOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_LINEAR_OPSRUNNER_ENABLE");
    isNormOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_NORM_OPSRUNNER_ENABLE");
    isMlpOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_MLP_OPSRUNNER_ENABLE");
    isPositionEmbeddingOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_POSITIONEMBEDDING_OPSRUNNER_ENABLE");
    isSelfAttentionKVCacheOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_SELFATTENTIONKVCACHE_OPSRUNNER_ENABLE");
    isSelfAttentionOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_SELFATTENTION_OPSRUNNER_ENABLE");
    isPositionEmbedding1dSplitOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_POSITIONEMBEDDING_1D_SPLIT_OPSRUNNER_ENABLE");
    isTransposeOpsRunnerEnable_ = IsEnable("ACLTRANSFORMER_TRANSPOSE_OPSRUNNER_ENABLE");
    isStreamSyncEveryKernelEnable_ = IsEnable("ACLTRANSFORMER_STREAM_SYNC_EVERY_KERNEL_ENABLE");
    isStreamSyncEveryRunnerEnable_ = IsEnable("ACLTRANSFORMER_STREAM_SYNC_EVERY_RUNNER_ENABLE");
    isStreamSyncEveryPlanEnable_ = IsEnable("ACLTRANSFORMER_STREAM_SYNC_EVERY_PLAN_ENABLE");
    isOpsRunnerSetupCacheEnable_ = IsEnable("ACLTRANSFORMER_OPSRUNNER_SETUP_CACHE_ENABLE");
    isOpsRunnerKernelCacheEnable_ = IsEnable("ACLTRANSFORMER_OPSRUNNER_KERNEL_CACHE_ENABLE");
    ASD_LOG(FATAL) << "Config:\nIsSaveTensor:" << isSaveTensor_
                   << "\nIsStreamSyncEveryRunnerEnable:" << isStreamSyncEveryRunnerEnable_
                   << "\nIsStreamSyncEveryKernelEnable:" << isStreamSyncEveryKernelEnable_
                   << "\nIsStreamSyncEveryPlanEnable:" << isStreamSyncEveryPlanEnable_
                   << "\nWorkspaceSize:" << workspaceSize_
                   << "\nIsOpsRunnerSetupCacheEnable:" << isOpsRunnerSetupCacheEnable_
                   << "\nIsOpsRunnerKernelCacheEnable:" << isOpsRunnerKernelCacheEnable_;
}

Config::~Config() {}

std::string Config::GetSaveTensorDir() { return "tensors"; }

bool Config::IsEnable(const char *env, bool enable)
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return enable;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsSaveTensor() { return isSaveTensor_; }

bool Config::IsAddOpsRunnerEnable() { return isAddOpsRunnerEnable_; }

bool Config::IsAddNormOpsRunnerEnable() { return isAddNormOpsRunnerEnable_; }

bool Config::IsRmsNormOpsRunnerEnable() { return isRmsNormOpsRunnerEnable_; }

bool Config::IsFfnOpsRunnerEnable() { return isFfnOpsRunnerEnable_; }

bool Config::IsLinearOpsRunnerEnable() { return isLinearOpsRunnerEnable_; }

bool Config::IsNormOpsRunnerEnable() { return isNormOpsRunnerEnable_; }

bool Config::IsMlpOpsRunnerEnable() { return isMlpOpsRunnerEnable_; }

bool Config::IsPositionEmbeddingOpsRunnerEnable() { return isPositionEmbeddingOpsRunnerEnable_; }

bool Config::IsPositionEmbedding1dSplitOpsRunnerEnable() { return isPositionEmbedding1dSplitOpsRunnerEnable_; }

bool Config::IsTransposeOpsRunnerEnable() { return isTransposeOpsRunnerEnable_; }

bool Config::IsSelfAttentionKVCacheOpsRunnerEnable() { return isSelfAttentionKVCacheOpsRunnerEnable_; }

bool Config::IsSelfAttentionOpsRunnerEnable() { return isSelfAttentionOpsRunnerEnable_; }

bool Config::IsStreamSyncEveryKernelEnable() { return isStreamSyncEveryKernelEnable_; }

bool Config::IsStreamSyncEveryRunnerEnable() { return isStreamSyncEveryRunnerEnable_; }

bool Config::IsStreamSyncEveryPlanEnable() { return isStreamSyncEveryPlanEnable_; }

bool Config::IsSkipKernel(const std::string &kernelName)
{
    if (skipKernelNames_.empty()) {
        return false;
    }
    if (skipKernelNames_.size() == 1 && skipKernelNames_.at(0) == "all") {
        return true;
    }
    for (auto &skipKernelName : skipKernelNames_) {
        if (AsdOps::StartsWith(kernelName, skipKernelName)) {
            return true;
        }
    }
    return false;
}

void Config::InitSkipKernelName()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_SKIP_KERNELS");
    if (!envStr) {
        return;
    }
    AsdOps::StrSplit(std::string(envStr), ',', skipKernelNames_);
}

uint64_t Config::GetWorkspaceSize() { return workspaceSize_; }

void Config::InitWorkspaceSize()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKSPACE_SIZE");
    if (!envStr) {
        return;
    }
    workspaceSize_ = atoll(envStr);
}

bool Config::Is910B() { return is910B_; }

void Config::InitIs910B()
{
    const int versionLen = 32;
    char version[versionLen] = {0};
    AsdRtDeviceGetSocVersion(version, versionLen);
    ASD_LOG(INFO) << "SocVersion:" << std::string(version);
    is910B_ = std::string(version).find("Ascend910B") != std::string::npos;
}

bool Config::IsOpsRunnerSetupCacheEnable() { return isOpsRunnerSetupCacheEnable_; }

bool Config::IsOpsRunnerKernelCacheEnable() { return isOpsRunnerKernelCacheEnable_; }
} // namespace AclTransformer