

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
#include "mlp_torch_runner.h"
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MlpTorchRunner::MlpTorchRunner(const MlpParam &param) : Runner("MlpTorchRunner"), param_(param) {}

MlpTorchRunner::~MlpTorchRunner() {}

AsdOps::Status MlpTorchRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    if (runnerVariantPack.inTensors.size() != 4) {
        return AsdOps::Status::FailStatus(1, "MlpTorchRunner inTensor num error!");
    }
#ifdef USE_TORCH_RUNNER
    ASD_LOG(INFO) << "MlpTorchRunner run!";
    ASD_LOG(INFO) << "start";
    at::Tensor hiddenStates = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[0]);
    at::Tensor weightGate = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[1]);
    at::Tensor weightDown = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[2]);
    at::Tensor weightUp = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[3]);
    ASD_LOG(INFO) << "hiddenStates" << hiddenStates.sizes();
    ASD_LOG(INFO) << "weightGate" << weightGate.sizes();
    ASD_LOG(INFO) << "weightDown" << weightDown.sizes();
    ASD_LOG(INFO) << "weightUp" << weightUp.sizes();
    // torch::save(hiddenStates.to(at::Device(at::kCPU)), "hiddenStates.pth");
    // torch::save(weightGate.to(at::Device(at::kCPU)), "weightGate.pth");
    // torch::save(weightDown.to(at::Device(at::kCPU)), "weightDown.pth");
    // torch::save(weightUp.to(at::Device(at::kCPU)), "weightUp.pth");
    hiddenStates = at::silu(at::matmul(hiddenStates, weightGate.t())) * at::matmul(hiddenStates, weightUp.t());
    at::Tensor atOutTensor = at::matmul(hiddenStates, weightDown.t()).contiguous();
    // torch::save(atOutTensor.to(at::Device(at::kCPU)), "mlp_llama_output.pth");
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, runnerVariantPack.outTensors[0]);
    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer