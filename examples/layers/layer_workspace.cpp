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
#include "layer_workspace.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"

namespace AclTransformer {

LayerWorkspace::LayerWorkspace() { SetWorkspace(AsdOps::GetSingleton<Config>().GetWorkspaceSize()); }

LayerWorkspace::~LayerWorkspace() { Free(); }

void LayerWorkspace::SetWorkspace(uint64_t workspaceSize)
{
    if (workspaceSize <= workspaceSize_) {
        ASD_LOG(INFO) << "LayerWorkspace::SetWorkspace workspaceSize:" << workspaceSize
                      << " <= workspaceSize_:" << workspaceSize_ << ", not new device mem";
        return;
    }

    ASD_LOG(INFO) << "LayerWorkspace::SetWorkspace AsdRtMemMallocDevice workspaceSize:" << workspaceSize;
    int st = AsdRtMemMallocDevice((void **)&workspace_, workspaceSize, ASDRT_MEM_DEFAULT);
    if (st != ASDRT_SUCCESS) {
        ASD_LOG(ERROR) << "LayerWorkspace::SetWorkspace AsdRtMemMallocDevice fail, ret:" << st;
        return;
    }
    workspaceSize_ = workspaceSize;
}

void *LayerWorkspace::GetWorkspace() { return workspace_; }

void LayerWorkspace::Free()
{
    if (workspace_) {
        ASD_LOG(INFO) << "LayerWorkspace::SetWorkspace AsdRtMemFreeDevice workspace:" << workspace_
                      << ", workspaceSize:" << workspaceSize_;
        AsdRtMemFreeDevice(workspace_);
        workspace_ = nullptr;
        workspaceSize_ = 0;
    }
}
} // namespace AclTransformer