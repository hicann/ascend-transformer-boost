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
#include "linear_torch_runner.h"
#include <ATen/ATen.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/utils/tensor_cache.h"
#include <torch_npu/csrc/framework/utils/OpPreparation.h>

namespace AclTransformer {
LinearTorchRunner::LinearTorchRunner(LinearParam &param) : Runner("LinearTorchRunner"), param_(param)
{
    ASD_LOG(INFO) << "LinearTorchRunner::LinearTorchRunner";
}

LinearTorchRunner::~LinearTorchRunner() {}

AsdOps::Status LinearTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    if (variantPack.inTensors.size() != 3) {
        return AsdOps::Status::FailStatus(1, "LinearTorchRunner inTensor num error!");
    }

    if (1) {
        at::Tensor *atInTensorA =
            AsdOps::GetSingleton<AclTransformer::TensorCache>().GetTensor(variantPack.inTensors[0].data);
        at::Tensor *atInTensorWeight =
            AsdOps::GetSingleton<AclTransformer::TensorCache>().GetTensor(variantPack.inTensors[1].data);
        at::Tensor *atInTensorWeightias =
            AsdOps::GetSingleton<AclTransformer::TensorCache>().GetTensor(variantPack.inTensors[2].data);
        at::Tensor atOutTensor = at::linear(*atInTensorA, *atInTensorWeight, *atInTensorWeightias).contiguous();
        ASD_LOG(INFO) << "LinearTorchRunner atOutTensor.format:"
                      << at_npu::native::CalcuOpUtil::GetTensorNpuFormat(atOutTensor);
        CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, variantPack.outTensors[0]);
    } else {
        at::Tensor atInTensorA = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[0]);
        at::Tensor atInTensorWeight = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[1]);
        at::Tensor atInTensorWeightias = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[2]);
        at::Tensor atOutTensor = at::linear(atInTensorA, atInTensorWeight, atInTensorWeightias).contiguous();
        ASD_LOG(INFO) << "LinearTorchRunner atOutTensor.format:"
                      << at_npu::native::CalcuOpUtil::GetTensorNpuFormat(atOutTensor);
        CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, variantPack.outTensors[0]);
    }

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer