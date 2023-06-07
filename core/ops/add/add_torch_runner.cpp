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
#include "add_torch_runner.h"
#include <ATen/ATen.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/utils/tensor_cache.h"

namespace AclTransformer {
AddTorchRunner::AddTorchRunner(const AddParam &param) : Runner("AddTorchRunner"), param_(param) {}

AddTorchRunner::~AddTorchRunner() {}

AsdOps::Status AddTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    at::Tensor atInTensorA = AsdOpsTensor2AtTensor(handle, variantPack.inTensors.at(0));
    at::Tensor atInTensorB = AsdOpsTensor2AtTensor(handle, variantPack.inTensors.at(1));

    at::Tensor atOutTensor = torch::add(atInTensorA, atInTensorB);
    CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, variantPack.outTensors[0]);
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer