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
#ifndef EXAMPLE_UTIL_H
#define EXAMPLE_UTIL_H
#include <vector>
#include <string>
#include <asdops/tensor.h>
#include <torch/torch.h>
#include "acltransformer/runner.h"
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "acltransformer/handle.h"
#include "acltransformer/variant_pack.h"

class ExampleUtil {
public:
    static void *GetCurrentStream();
    static int64_t GetTensorNpuFormat(const at::Tensor &tensor);
    static at::Tensor NpuFormatCast(const at::Tensor &tensor);
    static void BuildVariantPack(const std::vector<torch::Tensor> &inTensors,
                                 const std::vector<torch::Tensor> &outTensors,
                                 AclTransformer::VariantPack &variantPack);
    static AsdOps::Tensor AtTensor2AsdTensor(const at::Tensor &atTensor);
    static at::Tensor CreateAtTensorFromAsdOpsTensorDesc(const AsdOps::TensorDesc &tensorDesc);
    static void SaveTensor(const at::Tensor &tensor, const std::string &filePath);
    static void ContiguousAtTensor(std::vector<torch::Tensor> &atTensors);
};

#endif