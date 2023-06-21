

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
    ASD_LOG(INFO) << "RmsNormTorchRunner start";
    at::Tensor atInTensor = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[0]);
    at::Tensor atInTensorWeight = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[1]);
    ASD_LOG(INFO) << "receive inputs";

    caffe2::TypeMeta inTensorType = atInTensor.dtype();
    atInTensor = atInTensor.to(torch::kFloat32);
    // torch::save(atInTensor.to(at::Device(at::kCPU)), "atInTensor_cpp.pth");
    ASD_LOG(INFO) << "cast to f32";
    at::Tensor squareRslt = at::square(atInTensor);
    // torch::save(squareRslt.to(at::Device(at::kCPU)), "squareRslt_cpp.pth");
    at::Tensor variance = squareRslt.mean(-1, true);
    // torch::save(variance.to(at::Device(at::kCPU)), "variance_cpp.pth");
    ASD_LOG(INFO) << "square and mean ok";
    ASD_LOG(INFO) << "param_.rmsNormEps " << param_.rmsNormEps;
    ASD_LOG(INFO) << "atInTensor.shape " << atInTensor.sizes();
    ASD_LOG(INFO) << "variance.shape " << variance.sizes();
    // ASD_LOG(INFO) << "sum" <<  (variance + param_.rmsNormEps);
    at::Tensor addRslt = torch::add(variance, torch::tensor(param_.rmsNormEps));
    // torch::save(addRslt.to(at::Device(at::kCPU)), "addRslt_cpp.pth");
    at::Tensor rsqrtResult = at::rsqrt(addRslt);
    // torch::save(rsqrtResult.to(at::Device(at::kCPU)), "rsqrtResult_cpp.pth");
    ASD_LOG(INFO) << "rsqrt ok";
    at::Tensor hiddenStates = atInTensor * rsqrtResult;
    // torch::save(hiddenStates.to(at::Device(at::kCPU)), "mul_cpp.pth");
    ASD_LOG(INFO) << "mul ok";

    at::Tensor atOutTensor = atInTensorWeight * hiddenStates;
    // torch::save(atOutTensor.to(at::Device(at::kCPU)), "mul2_cpp.pth");
    ASD_LOG(INFO) << "mul 2 ok";
    atOutTensor = atOutTensor.to(inTensorType).contiguous();
    // torch::save(atOutTensor.to(at::Device(at::kCPU)), "atOutTensor_f32_cpp.pth");

    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, variantPack.outTensors[0]);
    ASD_LOG(INFO) << "RmsNormTorchRunner end";
    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer