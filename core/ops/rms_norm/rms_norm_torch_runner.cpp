

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
#include "rms_norm_torch_runner.h"
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
RmsNormTorchRunner::RmsNormTorchRunner(const RmsNormParam &param) : Runner("RmsNormTorchRunner"), param_(param) {}

RmsNormTorchRunner::~RmsNormTorchRunner() {}

AsdOps::Status RmsNormTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    if (variantPack.inTensors.size() != 2) {
        return AsdOps::Status::FailStatus(1, "RmsNormTorchRunner inTensor num error!");
    }
#ifdef USE_TORCH_RUNNER
    at::Tensor atInTensor = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[0]);
    at::Tensor atInTensorWeight = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[2]);

    caffe2::TypeMeta inTensorType = atInTensor.dtype();
    atInTensor = atInTensor.to(torch::kFloat32);
    at::Tensor variance = at::square(atInTensor).mean(-1);
    at::Tensor hiddenStates = hiddenStates * at::rsqrt(variance + param_.rmsNormEps);

    at::Tensor atOutTensor = atInTensorWeight * hiddenStates;
    atOutTensor = atOutTensor.to(inTensorType).contiguous();

    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, variantPack.outTensors[0]);
    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer