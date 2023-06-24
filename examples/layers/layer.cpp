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
#include "acltransformer/plan.h"
#include "acltransformer/plan_builder.h"
#include "examples/utils/example_util.h"
#include "layer_workspace.h"

namespace AclTransformer {
Layer::Layer(const std::string &layerName, const nlohmann::json &paramJson)
    : layerName_(layerName), paramJson_(paramJson)
{
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
    AsdOps::Status st = planBuilder.Build(opGraph_, plan_);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << opGraph_.name << " PlanBuilder build plan fail, error:" << st.Message();
        return;
    }
}

void Layer::Execute(AclTransformer::VariantPack &variantPack)
{
    void *stream = ExampleUtil::GetCurrentStream();
    if (lastStream_ != nullptr && lastStream_ != stream) {
        ASD_LOG(ERROR) << "stream changed";
        return;
    }
    lastStream_ = stream;
    AclTransformer::Handle handle = {stream};

    AsdOps::Status st = plan_.Setup(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << opGraph_.name << " Plan Setup fail error:" << st.Message();
        return;
    }

    variantPack.workspaceSize = plan_.GetWorkspaceSize();
    ASD_LOG(INFO) << opGraph_.name << " Plan GetWorkspaceSize:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        ASD_LOG(INFO) << opGraph_.name
                      << " AsdRtMemMallocDevice variantPack.workspaceSize:" << variantPack.workspaceSize;
        AsdOps::GetSingleton<LayerWorkspace>().SetWorkspace(variantPack.workspaceSize);
        variantPack.workspace = AsdOps::GetSingleton<LayerWorkspace>().GetWorkspace();
    }

    st = plan_.Execute(handle, variantPack);
    ASD_LOG_IF(!st.Ok(), ERROR) << opGraph_.name << " Plan Execute fail, error:" << st.Message();
}
} // namespace AclTransformer