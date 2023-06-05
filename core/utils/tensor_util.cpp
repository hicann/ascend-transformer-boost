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

at::Tensor AsdOpsTensor2AtTensor1(Handle handle, const AsdOps::Tensor &asdTensor)
{
    int32_t devId = 0;
    AsdRtDeviceGetCurrent(&devId);
    ASD_LOG(DEBUG) << "AsdRtDeviceGetCurrent devId:" << devId;
    at::TensorOptions options = at::TensorOptions().layout(torch::kStrided);
    if (asdTensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT) {
        options = options.dtype(at::kFloat);
    } else if (asdTensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT16) {
        options = options.dtype(at::kHalf);
    }

    at::Tensor rt;
    ASD_LOG(DEBUG) << "rt address before: " << asdTensor.data;
    rt = at::from_blob(asdTensor.data, IntArrayRef(asdTensor.desc.dims), options);
    ASD_LOG(DEBUG) << "rt address after: " << rt.data_ptr();
#ifdef TORCH_18
    rt = rt.to(at::Device(at::DeviceType::XLA, devId));
#else
    rt = rt.to(at::Device(at::kPrivateUse1, devId));
#endif
    ASD_LOG(DEBUG) << "rt address after to: " << rt.data_ptr();
    return rt;
}

at::Tensor AsdOpsTensor2AtTensor2(Handle handle, const AsdOps::Tensor &asdTensor)
{
    at::TensorOptions options = at::TensorOptions();
    if (asdTensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT) {
        options = options.dtype(at::kFloat);
    } else if (asdTensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT16) {
        options = options.dtype(at::kHalf);
    }

    at::Tensor newTensor = at::zeros(IntArrayRef(asdTensor.desc.dims), options);
#ifdef TORCH_18
    newTensor = newTensor.to(at::Device(at::DeviceType::XLA));
#else
    newTensor = newTensor.to(at::Device(at::kPrivateUse1));
#endif

    at::DataPtr dataPtr = at::DataPtr(asdTensor.data, newTensor.storage().device());
    newTensor.storage().set_data_ptr(std::move(dataPtr));
    ASD_LOG(INFO) << "set_data_ptr";
    newTensor = newTensor.contiguous();

    // ASD_LOG(INFO) << "AsdRtMemCopyAsync asdtensor to attensor";
    // int ret = AsdRtMemCopyAsync(newTensor.data_ptr(), asdTensor.dataSize, asdTensor.data, asdTensor.dataSize,
    //                             ASDRT_MEMCOPY_DEVICE_TO_DEVICE, handle.stream);
    // ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopyAsync fail";
    return newTensor;
}

AsdOps::Tensor AtTensor2AsdTensor(const at::Tensor &atTensor)
{
    at::Tensor contiguousAtTensor = atTensor.contiguous();
    ASD_LOG(INFO) << "contiguousAtTensor is contiguous:" << contiguousAtTensor.is_contiguous();
    AsdOps::Tensor asdTensor;
    asdTensor.desc.format = AsdOps::TENSOR_FORMAT_ND;
    asdTensor.data = contiguousAtTensor.storage().data_ptr().get();

    asdTensor.desc.dims.resize(contiguousAtTensor.sizes().size());
    for (uint64_t i = 0; i < contiguousAtTensor.sizes().size(); i++) {
        asdTensor.desc.dims[i] = contiguousAtTensor.sizes()[i];
    }

    auto it = DTYPE_MAP.find(contiguousAtTensor.scalar_type());
    if (it != DTYPE_MAP.end()) {
        asdTensor.desc.dtype = it->second;
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << contiguousAtTensor.scalar_type();
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
    at::Tensor newTensor =
        at::zeros(at::IntArrayRef(tensorDesc.dims.data(), tensorDesc.dims.size()), options);
#ifdef TORCH_18
    newTensor = newTensor.to(at::Device(at::DeviceType::XLA));
#else
    newTensor = newTensor.to(at::Device(at::kPrivateUse1));
#endif
    return newTensor;
}

at::Tensor AsdOpsTensor2AtTensor(Handle handle, const AsdOps::Tensor &asdTensor)
{
    return AsdOpsTensor2AtTensor2(handle, asdTensor);
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
} // namespace AclTransformer