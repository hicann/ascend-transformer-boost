/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "acltransformer/utils/tensor_cache.h"

namespace AclTransformer {
void TensorCache::AddTensor(void *data, at::Tensor *tensor) { tensorMap_[data] = tensor; }

at::Tensor *TensorCache::GetTensor(void *data)
{
    auto it = tensorMap_.find(data);
    if (it == tensorMap_.end()) {
        return nullptr;
    }
    return it->second;
}

void TensorCache::DeleteTensor(void *data)
{
    auto it = tensorMap_.find(data);
    if (it != tensorMap_.end()) {
        tensorMap_.erase(it);
    }
}
} // namespace AclTransformer