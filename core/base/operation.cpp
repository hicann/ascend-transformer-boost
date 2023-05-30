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
#include "acltransformer/operation.h"
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/runner.h"
#include "acltransformer/runner_builder.h"

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

AsdOps::Status Operation::Setup(Handle &handle, VariantPack &variantPack)
{
    if (runner_ == nullptr) {
        runner_ = CreateBestRunner(variantPack);
        if (runner_) {
            AsdOps::Status ret = runner_->Init();
            if (!ret.Ok()) {
                ASD_LOG(ERROR) << runner_->GetName() << " init fail";
                return ret;
            }
        } else {
            ASD_LOG(ERROR) << GetName() << " CreateBestRunner fail";
        }
    }

    if (runner_) {
        return runner_->Setup(handle, variantPack);
    }
    return AsdOps::Status::FailStatus(1, "runner is null");
}

uint64_t Operation::GetWorkspaceSize() { return runner_ ? runner_->GetWorkspaceSize() : 0; }

AsdOps::Status Operation::Execute(Handle &handle, VariantPack &variantPack)
{
    if (handle.stream == nullptr) {
        ASD_LOG(ERROR) << "handle.stream is null";
        return AsdOps::Status::FailStatus(1, "handle.stream is null");
    }
    if (!runner_) {
        ASD_LOG(ERROR) << "runner is null";
        return AsdOps::Status::FailStatus(1, "runner is null");
    }

    ASD_LOG(INFO) << runner_->GetName() << " execute start";
    AsdOps::Status st = runner_->Execute(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << runner_->GetName() << " execute fail, error:" << st.Message();
        return st;
    }

    ASD_LOG(INFO) << runner_->GetName() << " execute success";
    ASD_LOG(INFO) << "AsdRtStreamSynchronize stream:" << handle.stream;
    int ret = AsdRtStreamSynchronize(handle.stream);
    if (ret != 0) {
        ASD_LOG(ERROR) << "AsdRtStreamSynchronize fail";
        return AsdOps::Status::FailStatus(1, "AsdRtStreamSynchronize fail");
    }

    return AsdOps::Status::OkStatus();
}

Runner *Operation::CreateBestRunner(const VariantPack &variantPack)
{
    RunnerBuilder *runnerBuilder = FindBestRunnerBuilder(variantPack);
    if (runnerBuilder) {
        return runnerBuilder->Build();
    } else {
        ASD_LOG(ERROR) << "not find best runner builder";
        return nullptr;
    }
}
} // namespace AclTransformer