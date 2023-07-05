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
#ifndef ACLTRANSFORMER_TENSOR_UTIL_H
#define ACLTRANSFORMER_TENSOR_UTIL_H
#include <vector>
#include <string>
#include <asdops/tensor.h>
#include <asdops/run_info.h>
#include "acltransformer/variant_pack.h"
#include "acltransformer/handle.h"

namespace AclTransformer {
class TensorUtil {
public:
    static uint64_t CalcTensorDataSize(const AsdOps::Tensor &tensor);
    static uint64_t CalcTensorDataSize(const AsdOps::TensorDesc &tensorDesc);
    static std::string AsdOpsTensorToString(const AsdOps::Tensor &tensor);
    static std::string AsdOpsTensorDescToString(const AsdOps::TensorDesc &tensorDesc);
    static void SaveTensor(const AsdOps::Tensor &tensor, const std::string &filePath);
    static void SaveVariantPack(Handle &handle, const VariantPack &variantPack, const std::string &dirPath);
    static void SaveRunInfo(Handle &handle, const AsdOps::RunInfo &runInfo, const std::string &dirPath);
    static bool AsdOpsTensorDescEqual(const AsdOps::TensorDesc &tensorDescA, const AsdOps::TensorDesc &tensorDescB);
    static std::string AsdOpsDimsToString(const AsdOps::SVector<int64_t> &dims);
    static int64_t AlignInt(int64_t value, int align);
    static std::string AsdOpsRunInfoToString(const AsdOps::RunInfo &runInfo);
};
} // namespace AclTransformer
#endif