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
#include "acltransformer/utils/tensor_util.h"
#include <sstream>
#include <sys/stat.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/filesystem/filesystem.h>
#include "acltransformer/utils/tensor_cache.h"
#include <torch_npu/csrc/framework/utils/OpPreparation.h>

namespace AclTransformer {
static std::map<at::ScalarType, AsdOps::TensorDType> DTYPE_MAP = {
    {at::ScalarType::Bool, AsdOps::TENSOR_DTYPE_BOOL},   {at::ScalarType::Byte, AsdOps::TENSOR_DTYPE_UINT8},
    {at::ScalarType::Char, AsdOps::TENSOR_DTYPE_UINT8},  {at::ScalarType::Half, AsdOps::TENSOR_DTYPE_FLOAT16},
    {at::ScalarType::Float, AsdOps::TENSOR_DTYPE_FLOAT}, {at::ScalarType::Int, AsdOps::TENSOR_DTYPE_INT32},
    {at::ScalarType::Long, AsdOps::TENSOR_DTYPE_INT64},
};

void GetTensorDescs(const std::vector<AsdOps::Tensor> &tensors, AsdOps::SVector<AsdOps::TensorDesc> &tensorDescs)
{
    tensorDescs.resize(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        tensorDescs.at(i) = tensors.at(i).desc;
    }
}

uint64_t CalcTensorDataSize(const AsdOps::Tensor &tensor) { return CalcTensorDataSize(tensor.desc); }

uint64_t CalcTensorDataSize(const AsdOps::TensorDesc &tensorDesc)
{
    if (tensorDesc.dims.size() == 0) {
        return 0;
    }

    uint64_t dataItemSize = 0;
    switch (tensorDesc.dtype) {
    case AsdOps::TENSOR_DTYPE_BOOL: dataItemSize = sizeof(bool); break;
    case AsdOps::TENSOR_DTYPE_FLOAT: dataItemSize = sizeof(float); break;
    case AsdOps::TENSOR_DTYPE_FLOAT16: dataItemSize = 2; break;
    case AsdOps::TENSOR_DTYPE_INT8: dataItemSize = sizeof(int8_t); break;
    case AsdOps::TENSOR_DTYPE_INT16: dataItemSize = sizeof(int16_t); break;
    case AsdOps::TENSOR_DTYPE_INT32: dataItemSize = sizeof(int32_t); break;
    case AsdOps::TENSOR_DTYPE_INT64: dataItemSize = sizeof(int64_t); break;
    case AsdOps::TENSOR_DTYPE_UINT8: dataItemSize = sizeof(uint8_t); break;
    case AsdOps::TENSOR_DTYPE_UINT16: dataItemSize = sizeof(uint16_t); break;
    case AsdOps::TENSOR_DTYPE_UINT32: dataItemSize = sizeof(uint32_t); break;
    case AsdOps::TENSOR_DTYPE_UINT64: dataItemSize = sizeof(uint64_t); break;
    default: ASD_LOG(ERROR) << "not support dtype:" << tensorDesc.dtype;
    }

    int64_t elementCount = 1;
    for (auto i : tensorDesc.dims) {
        elementCount *= i;
    }

    return dataItemSize * elementCount;
}

static at::IntArrayRef IntArrayRef(const AsdOps::SVector<int64_t> &src)
{
    return at::IntArrayRef(src.data(), src.size());
}

at::Tensor AsdOpsTensor2AtCpuTensor(Handle handle, const AsdOps::Tensor &asdTensor)
{
    return AsdOpsTensor2AtTensor(handle, asdTensor).to(at::Device(at::kCPU)).contiguous();
}

AsdOps::Tensor AtTensor2AsdTensor(const at::Tensor &atTensor)
{
    ASD_LOG_IF(!atTensor.is_contiguous(), ERROR) << "atTensor is not contiguous";
    AsdOps::Tensor asdTensor;
    asdTensor.desc.format =
        static_cast<AsdOps::TensorFormat>(at_npu::native::CalcuOpUtil::GetTensorNpuFormat(atTensor));
    asdTensor.data = atTensor.data_ptr();

    asdTensor.desc.dims.resize(atTensor.sizes().size());
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        asdTensor.desc.dims[i] = atTensor.sizes()[i];
    }

    auto it = DTYPE_MAP.find(atTensor.scalar_type());
    if (it != DTYPE_MAP.end()) {
        asdTensor.desc.dtype = it->second;
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }

    asdTensor.dataSize = CalcTensorDataSize(asdTensor);

    return asdTensor;
}

at::Tensor CreateAtTensorFromAsdOpsTensorDesc(const AsdOps::TensorDesc &tensorDesc)
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
                  << ", format:" << at_npu::native::CalcuOpUtil::GetTensorNpuFormat(newTensor)
                  << ", is_contiguous:" << newTensor.is_contiguous();

    return newTensor;
}

at::Tensor AsdOpsTensor2AtTensor(Handle handle, const AsdOps::Tensor &asdTensor)
{
    at::Tensor newTensor = CreateAtTensorFromAsdOpsTensorDesc(asdTensor.desc);
    int ret = AsdRtMemCopy(newTensor.data_ptr(), asdTensor.dataSize, asdTensor.data, asdTensor.dataSize,
                           ASDRT_MEMCOPY_DEVICE_TO_DEVICE);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdOpsTensor2AtTensor AsdRtMemCopy fail";
    return newTensor;
}

at::Tensor AsdOpsTensor2AtTensorCache(Handle handle, const AsdOps::Tensor &asdTensor)
{
    static std::map<void *, at::Tensor> tensorCacheMap;
    auto it = tensorCacheMap.find(asdTensor.data);
    if (it != tensorCacheMap.end()) {
        ASD_LOG(INFO) << "use cache tensor, data:" << asdTensor.data;
        return it->second;
    }
    at::Tensor atTensor = AsdOpsTensor2AtTensor(handle, asdTensor);
    tensorCacheMap.insert(std::make_pair(asdTensor.data, atTensor));
    ASD_LOG(INFO) << "cache tensor, data:" << asdTensor.data;
    return atTensor;
}

void CopyAtTensor2AsdOpsTensor(void *stream, const at::Tensor &atTensor, AsdOps::Tensor &asdTensor)
{
    int ret = AsdRtStreamSynchronize(stream);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtStreamSynchronize AsdRtMemCopy fail";

    ret = AsdRtMemCopy(asdTensor.data, asdTensor.dataSize, atTensor.data_ptr(), asdTensor.dataSize,
                       ASDRT_MEMCOPY_DEVICE_TO_DEVICE);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail, atTensor.data:" << atTensor.data_ptr() << ", asdTensor.data"
                                << asdTensor.data;
}

std::string AsdOpsTensorToString(const AsdOps::Tensor &tensor)
{
    std::stringstream ss;
    ss << AsdOpsTensorDescToString(tensor.desc) << ", data:" << tensor.data << ", dataSize:" << tensor.dataSize;
    return ss.str();
}

void SaveVariantPack(Handle &handle, const VariantPack &variantPack, const std::string &dirPath)
{
    AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
        std::string fileName = "inTensor" + std::to_string(i) + ".pth";
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        at::Tensor *cachedTensor =
            AsdOps::GetSingleton<AclTransformer::TensorCache>().GetTensor(variantPack.inTensors.at(i).data);
        if (cachedTensor) {
            torch::save(cachedTensor->to(at::Device(at::kCPU)).contiguous(), filePath);
            ASD_LOG(INFO) << "save in tensor use cache";
        } else {
            at::Tensor atTensor = AsdOpsTensor2AtCpuTensor(handle, variantPack.inTensors.at(i));
            torch::save(atTensor, filePath);
        }
    }

    for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
        std::string fileName = "outTensor" + std::to_string(i) + ".pth";
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        at::Tensor *cachedTensor =
            AsdOps::GetSingleton<AclTransformer::TensorCache>().GetTensor(variantPack.outTensors.at(i).data);
        if (cachedTensor) {
            torch::save(cachedTensor->to(at::Device(at::kCPU)).contiguous(), filePath);
            ASD_LOG(INFO) << "save out tensor use cache";
        } else {
            at::Tensor atTensor = AsdOpsTensor2AtCpuTensor(handle, variantPack.outTensors.at(i));
            torch::save(atTensor, filePath);
        }
    }
}

std::string AsdOpsTensorDescToString(const AsdOps::TensorDesc &tensorDesc)
{
    std::stringstream ss;
    ss << "dtype:" << tensorDesc.dtype << ", format:" << tensorDesc.format << ", dims:[";
    for (size_t i = 0; i < tensorDesc.dims.size(); ++i) {
        if (i == 0) {
            ss << tensorDesc.dims.at(i);
        } else {
            ss << ", " << tensorDesc.dims.at(i);
        }
    }
    ss << "]";

    return ss.str();
}

bool AsdOpsTensorDescEqual(const AsdOps::TensorDesc &tensorDescA, const AsdOps::TensorDesc &tensorDescB)
{
    return tensorDescA.dims == tensorDescB.dims && tensorDescA.dtype == tensorDescB.dtype;
}

bool IsTensorDimEqual(const at::ArrayRef<long> &dims1, const AsdOps::SVector<int64_t> &dims2)
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