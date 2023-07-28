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
#include "transpose_torch_runner.h"
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
TransposeTorchRunner::TransposeTorchRunner(const TransposeParam &param) : Runner("TransposeTorchRunner"), param_(param)
{
}

TransposeTorchRunner::~TransposeTorchRunner() {}

AsdOps::Status TransposeTorchRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
#ifdef USE_TORCH_RUNNER
    at::Tensor atInTensorA = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors.at(0));
    ASD_LOG(INFO) << "in: " << atInTensorA.sizes();
    AsdOps::SVector<int64_t> perm;
    for (size_t i = 0; i < param_.perm.size(); i++) {
        perm.push_back(static_cast<int64_t>(param_.perm.at(i)));
    }
    at::Tensor atOutTensor =
        torch::permute(atInTensorA, at::IntArrayRef(perm.data(), perm.size())).contiguous();
    ASD_LOG(INFO) << "out: " << atOutTensor.sizes();
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, runnerVariantPack.outTensors[0]);
    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer