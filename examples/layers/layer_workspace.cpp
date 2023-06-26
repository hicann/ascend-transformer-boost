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
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "layer_workspace_rt.h"
#include "layer_workspace_torch.h"

namespace AclTransformer {

LayerWorkspace::LayerWorkspace()
{
    bool ret = IsUserTorch();
    ASD_LOG(FATAL) << "LayerWorkspace is use torch:" << IsUserTorch()
                   << ", can change it use ACLTRANSFORMER_WORKSPACE_USE_TORCH";

    if (ret) {
        base_.reset(new LayerWorkspaceTorch());
    } else {
        base_.reset(new LayerWorkspaceRt());
    }

    SetWorkspace(AsdOps::GetSingleton<Config>().GetWorkspaceSize());
}

LayerWorkspace::~LayerWorkspace() {}

void LayerWorkspace::SetWorkspace(uint64_t workspaceSize) { base_->SetWorkspace(workspaceSize); }

void *LayerWorkspace::GetWorkspace() { return base_->GetWorkspace(); }

bool LayerWorkspace::IsUserTorch()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKSPACE_USE_TORCH");
    if (envStr == nullptr) {
        return true;
    }
    return std::string(envStr) == "1";
}
} // namespace AclTransformer