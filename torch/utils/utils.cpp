
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
#include "utils.h"
#include <iostream>
#include <sys/stat.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma GCC diagnostic pop
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/filesystem/filesystem.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/config.h"
#include "acltransformer/utils/tensor_util.h"

void *Utils::GetCurrentStream()
{
    int32_t devId = 0;
    AsdRtDeviceGetCurrent(&devId);
    void *stream = c10_npu::getCurrentNPUStream(devId).stream();
    ASD_LOG_IF(stream == nullptr, ERROR) << "get current stream fail";
    return stream;
}

int64_t Utils::GetTensorNpuFormat(const at::Tensor &tensor)
{
#ifdef TORCH_GET_TENSOR_NPU_FORMAT_OLD
    return at_npu::native::CalcuOpUtil::get_tensor_npu_format(tensor);
#else
    return at_npu::native::CalcuOpUtil::GetTensorNpuFormat(tensor);
#endif
}

at::Tensor Utils::NpuFormatCast(const at::Tensor &tensor)
{
    return at_npu::native::NPUNativeFunctions::npu_format_cast(tensor, GetTensorNpuFormat(tensor));
}

void Utils::BuildVariantPack(const std::vector<torch::Tensor> &inTensors, const std::vector<torch::Tensor> &outTensors,
                             AclTransformer::VariantPack &variantPack)
{
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2AsdTensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2AsdTensor(outTensors.at(i)));
    }
}

AsdOps::Tensor Utils::AtTensor2AsdTensor(const at::Tensor &atTensor)
{
    static std::map<at::ScalarType, AsdOps::TensorDType> dtypeMap = {
        {at::ScalarType::Bool, AsdOps::TENSOR_DTYPE_BOOL},   {at::ScalarType::Byte, AsdOps::TENSOR_DTYPE_UINT8},
        {at::ScalarType::Char, AsdOps::TENSOR_DTYPE_INT8},   {at::ScalarType::Half, AsdOps::TENSOR_DTYPE_FLOAT16},
        {at::ScalarType::Float, AsdOps::TENSOR_DTYPE_FLOAT}, {at::ScalarType::Int, AsdOps::TENSOR_DTYPE_INT32},
        {at::ScalarType::Long, AsdOps::TENSOR_DTYPE_INT64},
    };

    ASD_LOG_IF(!atTensor.is_contiguous(), FATAL) << "atTensor is not contiguous";
    AsdOps::Tensor asdTensor;
    asdTensor.desc.format = static_cast<AsdOps::TensorFormat>(GetTensorNpuFormat(atTensor));
    asdTensor.data = atTensor.data_ptr();

    asdTensor.desc.dims.resize(atTensor.sizes().size());
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        asdTensor.desc.dims[i] = atTensor.sizes()[i];
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    if (it != dtypeMap.end()) {
        asdTensor.desc.dtype = it->second;
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }

    asdTensor.dataSize = AclTransformer::TensorUtil::CalcTensorDataSize(asdTensor);

    return asdTensor;
}

at::Tensor Utils::CreateAtTensorFromAsdOpsTensorDesc(const AsdOps::TensorDesc &tensorDesc)
{
    at::TensorOptions options = at::TensorOptions();
    if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_FLOAT) {
        options = options.dtype(at::kFloat);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_FLOAT16) {
        options = options.dtype(at::kHalf);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_BOOL) {
        options = options.dtype(at::kBool);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT64) {
        options = options.dtype(at::kLong);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT8) {
        options = options.dtype(at::kChar);
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << tensorDesc.dtype;
    }

#ifdef TORCH_18
    options = options.layout(torch::kStrided).requires_grad(false).device(at::DeviceType::XLA);
#else
    options = options.layout(torch::kStrided).requires_grad(false).device(at::kPrivateUse1);
#endif

    ASD_LOG(INFO) << "ApplyTensorWithFormat stat, format:" << tensorDesc.format;
    at::Tensor newTensor = at_npu::native::OpPreparation::ApplyTensorWithFormat(
        at::IntArrayRef(tensorDesc.dims.data(), tensorDesc.dims.size()), options, tensorDesc.format);
    ASD_LOG(INFO) << "ApplyTensorWithFormat end, newTensor.format:" << GetTensorNpuFormat(newTensor)
                  << ", is_contiguous:" << newTensor.is_contiguous();
    if (GetTensorNpuFormat(newTensor) != tensorDesc.format) {
        ASD_LOG(WARN) << "ApplyTensorWithFormat newTensor.format:" << GetTensorNpuFormat(newTensor)
                      << " != " << tensorDesc.format;
    }
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }

    ASD_LOG(INFO) << "ApplyTensorWithFormat success, newTensor.options:" << newTensor.options()
                  << ", format:" << GetTensorNpuFormat(newTensor) << ", is_contiguous:" << newTensor.is_contiguous();

    return newTensor;
}

void Utils::SaveTensor(const at::Tensor &tensor, const std::string &filePath)
{
    std::string dirPath = AsdOps::FileSystem::DirName(filePath);
    if (!AsdOps::FileSystem::Exists(dirPath)) {
        ASD_LOG(INFO) << "create dir:" << dirPath;
        AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    torch::save(tensor.to(at::Device(at::kCPU)), filePath);
}

void Utils::ContiguousAtTensor(std::vector<torch::Tensor> &atTensors)
{
    for (size_t i = 0; i < atTensors.size(); ++i) {
        if (!atTensors.at(i).is_contiguous()) {
            atTensors.at(i) = atTensors.at(i).contiguous();
        }
    }
}

void Utils::ContiguousAtTensor(torch::Tensor &atTensor)
{
    if (!atTensor.is_contiguous()) {
        atTensor = atTensor.contiguous();
    }
}