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
#ifndef ACLTRANSFORMER_TORCH_UTIL_H
#define ACLTRANSFORMER_TORCH_UTIL_H
#include <torch/torch.h>
#include <asdops/tensor.h>
#include "acltransformer/handle.h"

namespace AclTransformer {
class TorchUtil {
public:
    static int64_t GetTensorNpuFormat(const at::Tensor &tensor);
    static at::Tensor CreateAtTensorFromAsdOpsTensorDesc(const AsdOps::TensorDesc &tensorDesc);
    static at::Tensor AsdOpsTensor2AtTensor(Handle handle, const AsdOps::Tensor &asdTensor);
    static void CopyAtTensor2AsdOpsTensor(void *stream, const at::Tensor &tensor, AsdOps::Tensor &asdTensor);
    static at::Tensor AsdOpsTensor2AtCpuTensor(Handle handle, const AsdOps::Tensor &asdTensor);
    static bool IsTensorDimEqual(const at::ArrayRef<long> &dims1, const AsdOps::SVector<int64_t> &dims2);
};
} // namespace AclTransformer
#endif