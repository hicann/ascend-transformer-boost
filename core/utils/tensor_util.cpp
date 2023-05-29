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

namespace AclTransformer {
void GetTensorDescs(const std::vector<AsdOps::Tensor> &tensors, std::vector<AsdOps::TensorDesc> &tensorDescs)
{
    tensorDescs.resize(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        tensorDescs.at(i) = tensors.at(i).desc;
    }
}

uint64_t CalcTensorDataSize(const AsdOps::Tensor &tensor)
{
    uint64_t dataItemSize = 0;
    switch (tensor.desc.dtype) {
    case AsdOps::TENSOR_DTYPE_FLOAT: dataItemSize = sizeof(float); break;
    case AsdOps::TENSOR_DTYPE_FLOAT16: dataItemSize = 2; break;
    default: ASD_LOG(ERROR) << "not support dtype:" << tensor.desc.dtype;
    }

    return dataItemSize * tensor.Numel();
}

static at::IntArrayRef IntArrayRef(const AsdOps::SVector<int64_t> &src)
{
    return at::IntArrayRef(src.data(), src.size());
}

at::Tensor AsdOpsTensor2AtCpuTensor(const AsdOps::Tensor &asdTensor)
{
    at::TensorOptions options = at::TensorOptions().device(at::kCPU);
    if (asdTensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT) {
        options = options.dtype(at::kFloat);
    } else if (asdTensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT16) {
        options = options.dtype(at::kHalf);
    }

    torch::Tensor resultTensor = at::zeros(IntArrayRef(asdTensor.desc.dims), options);

#ifdef TORCH_18
    resultTensor = resultTensor.to(at::Device(at::DeviceType::XLA));
#else
    resultTensor = resultTensor.to(at::Device(at::kPrivateUse1));
#endif

    ASD_LOG(INFO) << "resultTensor.options:" << resultTensor.options()
                  << ", asdTensor.desc:" << AsdOpsTensorDescToString(asdTensor.desc);
    int st = AsdRtMemCopy(resultTensor.data_ptr(), asdTensor.dataSize, asdTensor.data, asdTensor.dataSize,
                          ASDRT_MEMCOPY_DEVICE_TO_DEVICE);
    ASD_LOG_IF(st != 0, ERROR) << "AsdRtMemCopy from device to host fail";

    resultTensor = resultTensor.to(at::Device(at::kCPU));
    return resultTensor;
}

at::Tensor AsdOpsTensor2AtTensor(const AsdOps::Tensor &asdTensor)
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

at::Tensor AsdOpsTensor2AtTensorCache(const AsdOps::Tensor &asdTensor)
{
    static std::map<void *, at::Tensor> tensorCacheMap;
    auto it = tensorCacheMap.find(asdTensor.data);
    if (it != tensorCacheMap.end()) {
        ASD_LOG(INFO) << "use cache tensor, data:" << asdTensor.data;
        return it->second;
    }
    at::Tensor atTensor = AsdOpsTensor2AtTensor(asdTensor);
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

void SaveVariantPack(const VariantPack &variantPack, const std::string &dirPath)
{
    AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
        std::string fileName = "inTensor" + std::to_string(i) + ".pth";
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        at::Tensor atTensor = AsdOpsTensor2AtCpuTensor(variantPack.inTensors.at(i));
        torch::save(atTensor, filePath);
    }

    for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
        std::string fileName = "outTensor" + std::to_string(i) + ".pth";
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        at::Tensor atTensor = AsdOpsTensor2AtCpuTensor(variantPack.outTensors.at(i));
        torch::save(atTensor, filePath);
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
} // namespace AclTransformer