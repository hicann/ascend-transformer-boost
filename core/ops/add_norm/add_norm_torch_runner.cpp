

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
#include "add_norm_torch_runner.h"
#include <ATen/ATen.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/utils/tensor_cache.h"

namespace AclTransformer {
AddNormTorchRunner::AddNormTorchRunner(const AddNormParam &param) : Runner("AddNormTorchRunner"), param_(param) {}

AddNormTorchRunner::~AddNormTorchRunner() {}

AsdOps::Status AddNormTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    if (variantPack.inTensors.size() != 4) {
        return AsdOps::Status::FailStatus(1, "AddNormTorchRunner inTensor num error!");
    }

    at::Tensor *atInTensorA = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[0].data);
    at::Tensor *atInTensorB = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[1].data);
    at::Tensor *atInTensorWeight = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[2].data);
    at::Tensor *atInTensorBias = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[3].data);
    at::Tensor *atOutTensor = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.outTensors[0].data);

    *atOutTensor = at::layer_norm(at::add(*atInTensorA, *atInTensorB), atInTensorWeight->sizes(), *atInTensorWeight,
                                  *atInTensorBias, param_.layerNormEps)
                       .contiguous();

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer