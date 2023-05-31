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

namespace AclTransformer {
AddTorchRunner::AddTorchRunner(const AddParam &param) : Runner("AddTorchRunner"), param_(param) {}

AddTorchRunner::~AddTorchRunner() {}

AsdOps::Status AddTorchRunner::Execute(Handle &handle, VariantPack &runInfo)
{
    ASD_LOG(INFO) << "AddTorchRunner::Execute start";
    at::Tensor atInTensorA = AsdOpsTensor2AtTensor(runInfo.inTensors[0]);
    at::Tensor atInTensorB = AsdOpsTensor2AtTensor(runInfo.inTensors[1]);

    at::Tensor addResultTensor = at::add(atInTensorA, atInTensorB).contiguous();
    int ret = AsdRtMemCopyAsync(runInfo.outTensors[0].data, runInfo.outTensors[0].dataSize,
                                addResultTensor.storage().data_ptr().get(), runInfo.outTensors[0].dataSize,
                                ASDRT_MEMCOPY_DEVICE_TO_DEVICE, handle.stream);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail";
    ASD_LOG(INFO) << "AddTorchRunner::Execute end";
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer