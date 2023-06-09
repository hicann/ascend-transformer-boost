

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
#include "add_norm_torch_runner.h"
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
AddNormTorchRunner::AddNormTorchRunner(const AddNormParam &param) : Runner("AddNormTorchRunner"), param_(param) {}

AddNormTorchRunner::~AddNormTorchRunner() {}

AsdOps::Status AddNormTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    if (variantPack.inTensors.size() != 4) {
        return AsdOps::Status::FailStatus(1, "AddNormTorchRunner inTensor num error!");
    }
#ifdef USE_TORCH_RUNNER
    at::Tensor atInTensorA = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[0]);
    at::Tensor atInTensorB = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[1]);
    at::Tensor atInTensorWeight = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[2]);
    at::Tensor atInTensorBias = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[3]);
    at::Tensor atOutTensor = at::layer_norm(at::add(atInTensorA, atInTensorB), atInTensorWeight.sizes(),
                                            atInTensorWeight, atInTensorBias, param_.layerNormEps)
                                 .contiguous();
    CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, variantPack.outTensors[0]);
    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer