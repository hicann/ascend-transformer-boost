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
#include "add_torch_runner.h"
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
AddTorchRunner::AddTorchRunner(const AddParam &param) : Runner("AddTorchRunner"), param_(param) {}

AddTorchRunner::~AddTorchRunner() {}

AsdOps::Status AddTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    ASD_LOG(INFO) << "scale : " << this->param_.scale;
#ifdef USE_TORCH_RUNNER
    at::Tensor atInTensorA = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors.at(0));
    at::Tensor atInTensorB = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors.at(1));

    atInTensorA = atInTensorA.to(torch::kFloat);
    at::Tensor atOutTensor = atInTensorA * this->param_.scale;
    atOutTensor = atOutTensor.to(torch::kHalf);

    atOutTensor = atOutTensor + atInTensorB;
    atOutTensor = atOutTensor.contiguous();
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, variantPack.outTensors[0]);
    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer