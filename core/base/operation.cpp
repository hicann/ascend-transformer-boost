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
#include "acltransformer/plan.h"

namespace AclTransformer {
Operation::Operation(const std::string &name) : name_(name) {}

Operation::~Operation()
{
    for (auto &it : runnerBuilders_) {
        delete it;
    }
    runnerBuilders_.clear();
}

std::string Operation::GetName() const { return name_; }

AsdOps::Status Operation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                     AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    uint64_t inTensorCount = GetInTensorCount();
    if (inTensors.size() != inTensorCount) {
        return AsdOps::Status::FailStatus(1, "inTensors size is not " + std::to_string(inTensorCount));
    }

    uint64_t outTensorCount = GetOutTensorCount();
    outTensorDescs.resize(outTensorCount);

    return InferShapeImpl(inTensors, outTensorDescs);
}

AsdOps::Status Operation::BuildPlan(Plan *plan)
{
    if (plan == nullptr) {
        return AsdOps::Status::FailStatus(1, "null plan");
    }

    plan->name_ = GetName() + "Plan";
    plan->runner_.reset(CreateBestRunner());
    return AsdOps::Status::OkStatus();
}

Runner *Operation::CreateBestRunner() const
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