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
#include "workspace.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "workspace_rt.h"
#include "workspace_torch.h"

namespace AclTransformer {

Workspace::Workspace()
{
    bool ret = IsUserTorch();
    ASD_LOG(INFO) << "Workspace is use torch:" << IsUserTorch()
                  << ", can change it use ACLTRANSFORMER_WORKSPACE_USE_TORCH";

    if (ret) {
        base_.reset(new WorkspaceTorch());
    } else {
        base_.reset(new WorkspaceRt());
    }

    SetWorkspace(AsdOps::GetSingleton<Config>().GetWorkspaceSize());
}

Workspace::~Workspace() {}

void Workspace::SetWorkspace(uint64_t workspaceSize) { base_->SetWorkspace(workspaceSize); }

void *Workspace::GetWorkspace() { return base_->GetWorkspace(); }

bool Workspace::IsUserTorch()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKSPACE_USE_TORCH");
    if (envStr == nullptr) {
        return true;
    }
    return std::string(envStr) == "1";
}
} // namespace AclTransformer