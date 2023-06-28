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
#include "layer.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/plan.h"
#include "acltransformer/plan_builder.h"
#include "acltransformer/statistic.h"
#include "examples/utils/example_util.h"
#include "examples/workspace/workspace.h"

namespace AclTransformer {
Layer::Layer(const std::string &layerName, const nlohmann::json &paramJson)
    : layerName_(layerName), paramJson_(paramJson)
{
    layerId_ = paramJson_["layerId"].get<int>();
}

Layer::~Layer()
{
    for (auto &node : opGraph_.nodes) {
        delete node.operation;
        node.operation = nullptr;
    }
}

std::string Layer::GetName() const { return layerName_; }

void Layer::BuildPlan()
{
    PlanBuilder planBuilder;
    std::string planName = layerName_ + "_" + std::to_string(layerId_);
    AsdOps::Status st = planBuilder.Build(planName, opGraph_, plan_);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << layerName_ << " PlanBuilder build plan fail, error:" << st.Message();
        return;
    }
}

void Layer::Execute(Handle &handle, AclTransformer::VariantPack &variantPack)
{
    AsdOps::Timer timer1;
    AsdOps::Status st = plan_.Setup(handle, variantPack);
    AsdOps::GetSingleton<Statistic>().planSetupTime = timer1.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << layerName_ << " Setup fail error:" << st.Message();
        return;
    }

    variantPack.workspaceSize = plan_.GetWorkspaceSize();
    ASD_LOG(INFO) << layerName_ << "  GetWorkspaceSize:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        ASD_LOG(INFO) << layerName_ << " AsdRtMemMallocDevice variantPack.workspaceSize:" << variantPack.workspaceSize;
        AsdOps::GetSingleton<Workspace>().SetWorkspace(variantPack.workspaceSize);
        variantPack.workspace = AsdOps::GetSingleton<Workspace>().GetWorkspace();
    }

    AsdOps::Timer timer2;
    st = plan_.Execute(handle, variantPack);
    AsdOps::GetSingleton<Statistic>().planExecuteTime = timer2.ElapsedMicroSecond();
    ASD_LOG_IF(!st.Ok(), ERROR) << layerName_ << " Execute fail, error:" << st.Message();
}
} // namespace AclTransformer