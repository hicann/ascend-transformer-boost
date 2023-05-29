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

namespace AclTransformer {
LinearTorchRunner::LinearTorchRunner(LinearParam &param) : Runner("LinearTorchRunner"), param_(param)
{
    ASD_LOG(INFO) << "LinearTorchRunner::LinearTorchRunner";
}

AsdOps::Status LinearTorchRunner::Init() { return AsdOps::Status::OkStatus(); }

AsdOps::Status LinearTorchRunner::Setup(Handle &handle, VariantPack &runInfo) { return AsdOps::Status::OkStatus(); }

uint64_t LinearTorchRunner::GetWorkspaceSize() { return 0; }

AsdOps::Status LinearTorchRunner::Execute(Handle &handle, VariantPack &runInfo)
{
    if (runInfo.inTensors.size() != 3) {
        return AsdOps::Status::FailStatus(1, "LinearTorchRunner inTensor num error!");
    }
    at::Tensor atInTensorA = AsdOpsTensor2AtTensor(runInfo.inTensors[0]);
    at::Tensor atInTensorWeight = AsdOpsTensor2AtTensorCache(runInfo.inTensors[1]);
    at::Tensor atInTensorWeightias = AsdOpsTensor2AtTensorCache(runInfo.inTensors[2]);
    at::Tensor outputTensor = at::linear(atInTensorA, atInTensorWeight, atInTensorWeightias).contiguous();
    int ret = AsdRtMemCopyAsync(runInfo.outTensors[0].data, runInfo.outTensors[0].dataSize,
                                outputTensor.storage().data_ptr().get(), runInfo.outTensors[0].dataSize,
                                ASDRT_MEMCOPY_DEVICE_TO_DEVICE, handle.stream);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail";
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer