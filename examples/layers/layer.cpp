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

void Layer::SetWorkspace(uint64_t workspaceSize)
{
    if (workspaceSize <= workspaceSize_) {
        ASD_LOG(INFO) << GetName() << " workspaceSize:" << workspaceSize << " <= workspaceSize_:" << workspaceSize_
                      << ", not new device mem";
        return;
    }

    if (workspace_) {
        ASD_LOG(INFO) << GetName() << " AsdRtMemFreeDevice workspace:" << workspace_
                      << ", workspaceSize:" << workspaceSize_;
        AsdRtMemFreeDevice(workspace_);
        workspace_ = nullptr;
        workspaceSize_ = 0;
    }

    ASD_LOG(INFO) << GetName() << " AsdRtMemMallocDevice workspaceSize:" << workspaceSize;
    int st = AsdRtMemMallocDevice((void **)&workspace_, workspaceSize, ASDRT_MEM_DEFAULT);
    if (st != ASDRT_SUCCESS) {
        ASD_LOG(ERROR) << GetName() << " AsdRtMemMallocDevice fail, ret:" << st;
        return;
    }
    workspaceSize_ = workspaceSize;
}

void Layer::ExecuteOperationGraph(AclTransformer::OperationGraph &opGraph, AclTransformer::VariantPack &variantPack)
{
    void *stream = ExampleUtil::GetCurrentStream();
    if (lastStream_ != nullptr && lastStream_ != stream) {
        ASD_LOG(ERROR) << "stream changed";
        return;
    }
    lastStream_ = stream;
    AclTransformer::Handle handle = {stream};

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
        SetWorkspace(variantPack.workspaceSize);
        variantPack.workspace = workspace_;
    }

    st = plan.Execute(handle, variantPack);
    ASD_LOG_IF(!st.Ok(), ERROR) << opGraph.name << " Plan Execute fail, error:" << st.Message();
}
} // namespace AclTransformer