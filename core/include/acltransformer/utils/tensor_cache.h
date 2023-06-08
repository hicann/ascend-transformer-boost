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
#ifndef ACLTRANSFOERM_TENSOR_CACHE_H
#define ACLTRANSFOERM_TENSOR_CACHE_H
#include <map>
#include <torch/torch.h>
#include <asdops/utils/singleton/singleton.h>

namespace AclTransformer {
class TensorCache {
public:
    void AddTensor(void *data, at::Tensor *tensor);
    at::Tensor *GetTensor(void *data);
    void DeleteTensor(void *data);

private:
    std::map<void *, at::Tensor *> tensorMap_;
};
} // namespace AclTransformer
#endif