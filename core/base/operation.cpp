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
#include "acltransformer/operation.h"
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/runner.h"
#include "acltransformer/runner_builder.h"
#include "acltransformer/config.h"
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
Operation::Operation(const std::string &name) : name_(name) {}

Operation::~Operation()
{
    for (auto &it : runnerBuilders_) {
        delete it;
    }
    runnerBuilders_.clear();

    if (runner_) {
        delete runner_;
        runner_ = nullptr;
    }
}

std::string Operation::GetName() const { return name_; }

AsdOps::Status Operation::Setup(VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " Setup variantPack:" << variantPack.ToString();
    if (runner_ == nullptr) {
        runner_ = CreateBestRunner();
        ASD_LOG_IF(runner_ == nullptr, ERROR) << GetName() << " CreateBestRunner fail";
    }
    if (!runner_) {
        return AsdOps::Status::FailStatus(1, "create best runner fail");
    }

    ASD_LOG(INFO) << runner_->GetName() << " Setup start";
    AsdOps::Status st = runner_->Setup(variantPack);
    ASD_LOG(INFO) << runner_->GetName() << " Setup end, result:" << st.Message();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << GetName() << " Setup fail, error:" << st.Message();
        return st;
    }

    ASD_LOG(INFO) << GetName() << " Setup success";
    return AsdOps::Status::OkStatus();
}

uint64_t Operation::GetWorkspaceSize() { return runner_ ? runner_->GetWorkspaceSize() : 0; }

AsdOps::Status Operation::Execute(Handle &handle, VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " Execute variantPack:" << variantPack.ToString();
    if (handle.stream == nullptr) {
        ASD_LOG(ERROR) << GetName() << " handle.stream is null";
        return AsdOps::Status::FailStatus(1, "handle.stream is null");
    }
    if (!runner_) {
        ASD_LOG(ERROR) << GetName() << " runner is null";
        return AsdOps::Status::FailStatus(1, "runner is null");
    }

    ASD_LOG(INFO) << runner_->GetName() << " Execute start";
    AsdOps::Status st = runner_->Execute(handle, variantPack);
    ASD_LOG(INFO) << runner_->GetName() << " Execute end";
    if (!st.Ok()) {
        ASD_LOG(ERROR) << GetName() << " execute fail, error:" << st.Message();
        return st;
    }

    if (AsdOps::GetSingleton<Config>().IsStreamSyncEveryOperationEnable()) {
        int ret = AsdRtStreamSynchronize(handle.stream);
        if (ret != 0) {
            ASD_LOG(ERROR) << GetName() << " AsdRtStreamSynchronize fail, ret:" << ret;
            return AsdOps::Status::FailStatus(1, "AsdRtStreamSynchronize fail");
        }
    }

    if (AsdOps::GetSingleton<Config>().IsSaveTensor()) {
        std::string dirPath = Config::GetSaveTensorDir() + "/" + runner_->GetName();
        TensorUtil::SaveVariantPack(handle, variantPack, dirPath);
        ASD_LOG(INFO) << GetName() << " SaveVariantPack " << dirPath;
    }

    ASD_LOG(INFO) << GetName() << " Execute success";
    return AsdOps::Status::OkStatus();
}

Runner *Operation::CreateBestRunner()
{
    RunnerBuilder *runnerBuilder = FindBestRunnerBuilder();
    if (runnerBuilder) {
        return runnerBuilder->Build();
    } else {
        ASD_LOG(ERROR) << GetName() << " not find best runner builder";
        return nullptr;
    }
}
} // namespace AclTransformer