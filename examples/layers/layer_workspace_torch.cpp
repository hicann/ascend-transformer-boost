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
#include "layer_workspace_torch.h"
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"

namespace AclTransformer {
LayerWorkspaceTorch::LayerWorkspaceTorch() { ASD_LOG(INFO) << "LayerWorkspaceTorch::LayerWorkspaceTorch called"; }

LayerWorkspaceTorch::~LayerWorkspaceTorch() {}

void LayerWorkspaceTorch::SetWorkspace(uint64_t workspaceSize)
{
    if (workspaceSize <= workspaceSize_) {
        ASD_LOG(INFO) << "LayerWorkspaceTorch::SetWorkspace workspaceSize:" << workspaceSize
                      << " <= workspaceSize_:" << workspaceSize_ << ", not new device mem";
        return;
    }

    at::TensorOptions options = at::TensorOptions();
    options = options.dtype(at::kByte);

#ifdef TORCH_18
    options = options.layout(torch::kStrided).requires_grad(false).device(at::DeviceType::XLA);
#else
    options = options.layout(torch::kStrided).requires_grad(false).device(at::kPrivateUse1);
#endif

    at::SmallVector<int64_t, 8> dim = {1024, 1024, (int64_t)workspaceSize / 1024 / 1024};
    ASD_LOG(FATAL) << "LayerWorkspaceTorch ApplyTensorWithFormat workspaceSize:" << workspaceSize << ", dim:" << dim;
    workspaceTensor_ = at_npu::native::OpPreparation::ApplyTensorWithFormat(dim, options, 2);
    ASD_LOG(FATAL) << "LayerWorkspaceTorch ApplyTensorWithFormat success";
    workspaceSize_ = workspaceSize;
}

void *LayerWorkspaceTorch::GetWorkspace() { return workspaceTensor_.data_ptr(); }
} // namespace AclTransformer