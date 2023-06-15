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
#include "acltransformer/torch/torch_util.h"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>

namespace AclTransformer {
int64_t TorchUtil::GetTensorNpuFormat(const at::Tensor &tensor)
{
#ifdef TORCH_GET_TENSOR_NPU_FORMAT_OLD
    return at_npu::native::CalcuOpUtil::get_tensor_npu_format(tensor);
#else
    return at_npu::native::CalcuOpUtil::GetTensorNpuFormat(tensor);
#endif
}

at::Tensor TorchUtil::NpuFormatCast(const at::Tensor &tensor)
{
    return at_npu::native::NPUNativeFunctions::npu_format_cast(tensor, TorchUtil::GetTensorNpuFormat(tensor));
}

void *TorchUtil::GetTensorDataPtr(const at::Tensor &tensor)
{
    ASD_LOG(INFO) << "tensor.storage().unsafeGetStorageImpl()->data():"
                  << tensor.storage().unsafeGetStorageImpl()->data()
                  << ", tensor.storage_offset():" << tensor.storage_offset()
                  << ", tensor.itemsize():" << tensor.itemsize() << ", tensor.data_ptr():" << tensor.data_ptr();
    return tensor.storage().unsafeGetStorageImpl()->data() + tensor.storage_offset() * tensor.itemsize();
}

at::Tensor TorchUtil::CreateAtTensorFromAsdOpsTensorDesc(const AsdOps::TensorDesc &tensorDesc)
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
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << tensorDesc.dtype;
    }

#ifdef TORCH_18
    options = options.layout(torch::kStrided).requires_grad(false).device(at::DeviceType::XLA);
#else
    options = options.layout(torch::kStrided).requires_grad(false).device(at::kPrivateUse1);
#endif

    ASD_LOG(INFO) << "ApplyTensorWithFormat stat, format:" << tensorDesc.format;
    at::Tensor newTensor =
        at_npu::native::OpPreparation::ApplyTensorWithFormat(
            at::IntArrayRef(tensorDesc.dims.data(), tensorDesc.dims.size()), options, tensorDesc.format)
            .contiguous();
    ASD_LOG(INFO) << "ApplyTensorWithFormat success, newTensor.options:" << newTensor.options()
                  << ", format:" << TorchUtil::GetTensorNpuFormat(newTensor)
                  << ", is_contiguous:" << newTensor.is_contiguous();

    return newTensor;
}

at::Tensor TorchUtil::AsdOpsTensor2AtTensor(Handle handle, const AsdOps::Tensor &asdTensor)
{
    at::Tensor newTensor = CreateAtTensorFromAsdOpsTensorDesc(asdTensor.desc);
    int ret = AsdRtMemCopy(newTensor.data_ptr(), asdTensor.dataSize, asdTensor.data, asdTensor.dataSize,
                           ASDRT_MEMCOPY_DEVICE_TO_DEVICE);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdOpsTensor2AtTensor AsdRtMemCopy fail, ret:" << ret;
    return newTensor;
}

void TorchUtil::CopyAtTensor2AsdOpsTensor(void *stream, const at::Tensor &atTensor, AsdOps::Tensor &asdTensor)
{
    ASD_LOG(INFO) << "CopyAtTensor2AsdOpsTensor, asdTensor.dataSize:" << asdTensor.dataSize;
    static std::map<at::ScalarType, AsdOps::TensorDType> dtypeMap = {
        {at::ScalarType::Bool, AsdOps::TENSOR_DTYPE_BOOL},   {at::ScalarType::Byte, AsdOps::TENSOR_DTYPE_UINT8},
        {at::ScalarType::Char, AsdOps::TENSOR_DTYPE_UINT8},  {at::ScalarType::Half, AsdOps::TENSOR_DTYPE_FLOAT16},
        {at::ScalarType::Float, AsdOps::TENSOR_DTYPE_FLOAT}, {at::ScalarType::Int, AsdOps::TENSOR_DTYPE_INT32},
        {at::ScalarType::Long, AsdOps::TENSOR_DTYPE_INT64},
    };

    AsdOps::TensorDType dtype = AsdOps::TENSOR_DTYPE_UNDEFINED;
    auto it = dtypeMap.find(atTensor.scalar_type());
    if (it != dtypeMap.end()) {
        dtype = it->second;
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }
    ASD_LOG_IF(dtype != asdTensor.desc.dtype, ERROR)
        << "atTensor dtype:" << dtype << " != asdTensor.dtype:" << asdTensor.desc.dtype;

    ASD_LOG_IF(!atTensor.is_contiguous(), ERROR) << "atTensor is not is_contiguous, can't copy to asdTensor";
    c10_npu::NPUStream npuStream = c10_npu::getCurrentNPUStream();

    ASD_LOG_IF(atTensor.numel() * atTensor.element_size() != asdTensor.dataSize, ERROR)
        << " not atTensor.numel() * atTensor.element_size():" << atTensor.numel() * atTensor.element_size()
        << ", asdTensor.dataSize:" << asdTensor.dataSize;

    at::Tensor newTensor =
        at_npu::native::NPUNativeFunctions::npu_format_cast(atTensor, TorchUtil::GetTensorNpuFormat(atTensor));

    ASD_LOG(INFO) << "npuStream.synchronize";
    npuStream.synchronize();

    ASD_LOG_IF(GetTensorDataPtr(newTensor) != newTensor.data_ptr(), ERROR) << " newTensor.data_ptr() is not equal";

    int ret = AsdRtMemCopy(asdTensor.data, asdTensor.dataSize, newTensor.data_ptr(), asdTensor.dataSize,
                           ASDRT_MEMCOPY_DEVICE_TO_DEVICE);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail, newTensor.data:" << newTensor.data_ptr() << ", asdTensor.data"
                                << asdTensor.data;
}

at::Tensor TorchUtil::AsdOpsTensor2AtCpuTensor(Handle handle, const AsdOps::Tensor &asdTensor)
{
    return AsdOpsTensor2AtTensor(handle, asdTensor).to(at::Device(at::kCPU)).contiguous();
}

bool TorchUtil::IsTensorDimEqual(const at::ArrayRef<long> &dims1, const AsdOps::SVector<int64_t> &dims2)
{
    if (dims1.size() != dims2.size()) {
        return false;
    }
    for (size_t i = 0; i < dims1.size(); ++i) {
        if (dims1.at(i) != dims2.at(i)) {
            return false;
        }
    }
    return true;
}
} // namespace AclTransformer