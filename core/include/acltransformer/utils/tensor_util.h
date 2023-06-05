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
#ifndef ACLTRANSFORMER_TENSOR_UTIL_H
#define ACLTRANSFORMER_TENSOR_UTIL_H
#include <vector>
#include <string>
#include <asdops/tensor.h>
#include <torch/torch.h>
#include "acltransformer/variant_pack.h"
#include "acltransformer/handle.h"

namespace AclTransformer {
void GetTensorDescs(const std::vector<AsdOps::Tensor> &tensors, AsdOps::SVector<AsdOps::TensorDesc> &tensorDescs);
uint64_t CalcTensorDataSize(const AsdOps::Tensor &tensor);
uint64_t CalcTensorDataSize(const AsdOps::TensorDesc &tensorDesc);
at::Tensor CreateAtTensorFromAsdOpsTensorDesc(const AsdOps::TensorDesc &tensorDesc);
at::Tensor AsdOpsTensor2AtTensor(Handle handle, const AsdOps::Tensor &asdTensor);
at::Tensor AsdOpsTensor2AtTensorCache(Handle handle, const AsdOps::Tensor &asdTensor);
at::Tensor AsdOpsTensor2AtCpuTensor(Handle handle, const AsdOps::Tensor &asdTensor);
std::string AsdOpsTensorToString(const AsdOps::Tensor &tensor);
std::string AsdOpsTensorDescToString(const AsdOps::TensorDesc &tensorDesc);
AsdOps::Tensor AtTensor2AsdTensor(const at::Tensor &atTensor);
void SaveVariantPack(Handle &handle, const VariantPack &variantPack, const std::string &dirPath);
bool AsdOpsTensorDescEqual(const AsdOps::TensorDesc &tensorDescA, const AsdOps::TensorDesc &tensorDescB);
} // namespace AclTransformer
#endif