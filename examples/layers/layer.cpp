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
#include "acltransformer/plan.h"
#include "acltransformer/plan_builder.h"
#include "examples/utils/example_util.h"

namespace AclTransformer {
Layer::Layer(const std::string &layerName) : layerName_(layerName) {}

Layer::~Layer() {}

std::string Layer::GetName() const { return layerName_; }

void Layer::SetParam(const nlohmann::json &paramJson) { paramJson_ = paramJson; }

void Layer::ExecuteOperationGraph(AclTransformer::OperationGraph &opGraph, AclTransformer::VariantPack &variantPack)
{
    AclTransformer::Handle handle = {ExampleUtil::GetCurrentStream()};

    AclTransformer::PlanBuilder planBuilder;
    AclTransformer::Plan plan;
    AsdOps::Status st = planBuilder.Build(variantPack, opGraph, plan);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << opGraph.name << " PlanBuilder build plan fail, error:" << st.Message();
        return;
    }

    st = plan.Setup(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << opGraph.name << " Plan Setup fail error:" << st.Message();
        return;
    }

    variantPack.workspaceSize = plan.GetWorkspaceSize();
    ASD_LOG(INFO) << opGraph.name << " Plan GetWorkspaceSize:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        ASD_LOG(INFO) << opGraph.name
                      << " AsdRtMemMallocDevice variantPack.workspaceSize:" << variantPack.workspaceSize;
        int st = AsdRtMemMallocDevice((void **)&variantPack.workspace, variantPack.workspaceSize, ASDRT_MEM_DEFAULT);
        if (st != ASDRT_SUCCESS) {
            ASD_LOG(ERROR) << opGraph.name << " AsdRtMemMallocDevice fail";
            return;
        }
    }

    st = plan.Execute(handle, variantPack);
    ASD_LOG_IF(!st.Ok(), ERROR) << opGraph.name << " Plan Execute fail, error:" << st.Message();

    if (variantPack.workspace != nullptr) {
        AsdRtMemFreeDevice(variantPack.workspace);
        ASD_LOG(INFO) << opGraph.name << " AsdRtMemFreeDevice free:" << variantPack.workspace;
        variantPack.workspace = nullptr;
        variantPack.workspaceSize = 0;
    }
}
} // namespace AclTransformer