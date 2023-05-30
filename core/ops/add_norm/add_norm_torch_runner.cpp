

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

namespace AclTransformer {
AddNormTorchRunner::AddNormTorchRunner(const AddNormParam &param) : Runner("AddNormTorchRunner"), param_(param) {}

AsdOps::Status AddNormTorchRunner::Init() { return AsdOps::Status::OkStatus(); }

AsdOps::Status AddNormTorchRunner::Setup(Handle &handle, VariantPack &runInfo) { return AsdOps::Status::OkStatus(); }

uint64_t AddNormTorchRunner::GetWorkspaceSize() { return 0; }

AsdOps::Status AddNormTorchRunner::Execute(Handle &handle, VariantPack &runInfo)
{
    if (runInfo.inTensors.size() != 4) {
        return AsdOps::Status::FailStatus(1, "AddNormTorchRunner inTensor num error!");
    }

    at::Tensor atInTensorA = AsdOpsTensor2AtTensor(runInfo.inTensors[0]);
    at::Tensor atInTensorB = AsdOpsTensor2AtTensor(runInfo.inTensors[1]);
    at::Tensor atInTensorWeight = AsdOpsTensor2AtTensorCache(runInfo.inTensors[2]);
    at::Tensor atInTensorBias = AsdOpsTensor2AtTensorCache(runInfo.inTensors[3]);
    at::Tensor addResultTensor = at::add(atInTensorA, atInTensorB);
    const double eps = 1e-12;
    at::Tensor outputTensor =
        at::layer_norm(addResultTensor, atInTensorWeight.sizes(), atInTensorWeight, atInTensorBias, eps).contiguous();
    int ret = AsdRtMemCopyAsync(runInfo.outTensors[0].data, runInfo.outTensors[0].dataSize,
                                outputTensor.storage().data_ptr().get(), runInfo.outTensors[0].dataSize,
                                ASDRT_MEMCOPY_DEVICE_TO_DEVICE, handle.stream);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail";
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer