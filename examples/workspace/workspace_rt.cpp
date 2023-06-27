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
#include "workspace_rt.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>

namespace AclTransformer {

WorkspaceRt::WorkspaceRt() { ASD_LOG(INFO) << "WorkspaceRt::WorkspaceRt called"; }

WorkspaceRt::~WorkspaceRt() { Free(); }

void WorkspaceRt::SetWorkspace(uint64_t workspaceSize)
{
    if (workspaceSize <= workspaceSize_) {
        ASD_LOG(INFO) << "WorkspaceRt::SetWorkspace workspaceSize:" << workspaceSize
                      << " <= workspaceSize_:" << workspaceSize_ << ", not new device mem";
        return;
    }

    Free();

    ASD_LOG(INFO) << "WorkspaceRt::SetWorkspace AsdRtMemMallocDevice workspaceSize:" << workspaceSize;
    int st = AsdRtMemMallocDevice((void **)&workspace_, workspaceSize, ASDRT_MEM_DEFAULT);
    if (st != ASDRT_SUCCESS) {
        ASD_LOG(ERROR) << "WorkspaceRt::SetWorkspace AsdRtMemMallocDevice fail, ret:" << st;
        return;
    }
    workspaceSize_ = workspaceSize;
}

void *WorkspaceRt::GetWorkspace() { return workspace_; }

void WorkspaceRt::Free()
{
    if (workspace_) {
        ASD_LOG(INFO) << "WorkspaceRt::SetWorkspace AsdRtMemFreeDevice workspace:" << workspace_
                      << ", workspaceSize:" << workspaceSize_;
        AsdRtMemFreeDevice(workspace_);
        workspace_ = nullptr;
        workspaceSize_ = 0;
    }
}
} // namespace AclTransformer