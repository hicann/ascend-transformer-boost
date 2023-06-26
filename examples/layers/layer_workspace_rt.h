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
#ifndef LAYER_WORKSPACE_RT_H
#define LAYER_WORKSPACE_RT_H
#include "layer_workspace_base.h"

namespace AclTransformer {
class LayerWorkspaceRt : public LayerWorkspaceBase {
public:
    LayerWorkspaceRt();
    virtual ~LayerWorkspaceRt();
    void SetWorkspace(uint64_t workspaceSize) override;
    void *GetWorkspace() override;

private:
    void Free();

private:
    void *workspace_ = nullptr;
    uint64_t workspaceSize_ = 0;
};
} // namespace AclTransformer
#endif