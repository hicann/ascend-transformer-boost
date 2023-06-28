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
#include "acltransformer/utils/tensor_util.h"
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <asdops/utils/binfile/binfile.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/filesystem/filesystem.h>

namespace AclTransformer {
const char *TENSOR_FILE_NAME_EXT = ".bin";

uint64_t TensorUtil::CalcTensorDataSize(const AsdOps::Tensor &tensor) { return CalcTensorDataSize(tensor.desc); }

uint64_t TensorUtil::CalcTensorDataSize(const AsdOps::TensorDesc &tensorDesc)
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

std::string TensorUtil::AsdOpsTensorToString(const AsdOps::Tensor &tensor)
{
    std::stringstream ss;
    ss << AsdOpsTensorDescToString(tensor.desc) << ", data:" << tensor.data << ", dataSize:" << tensor.dataSize;
    return ss.str();
}

std::string TensorUtil::AsdOpsTensorDescToString(const AsdOps::TensorDesc &tensorDesc)
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

bool TensorUtil::AsdOpsTensorDescEqual(const AsdOps::TensorDesc &tensorDescA, const AsdOps::TensorDesc &tensorDescB)
{
    return tensorDescA.dtype == tensorDescB.dtype && tensorDescA.format == tensorDescB.format &&
           tensorDescA.dims == tensorDescB.dims;
}

void TensorUtil::SaveTensor(const AsdOps::Tensor &tensor, const std::string &filePath)
{
    ASD_LOG(INFO) << "save asdtensor start, tensor:" << AsdOpsTensorToString(tensor) << ", filePath:" << filePath;
    AsdOps::BinFile binFile;
    binFile.AddAttr("format", std::to_string(tensor.desc.format));
    binFile.AddAttr("dtype", std::to_string(tensor.desc.dtype));
    binFile.AddAttr("dims", AsdOpsDimsToString(tensor.desc.dims));
    if (tensor.data) {
        std::vector<char> hostData(tensor.dataSize);
        int st =
            AsdRtMemCopy(hostData.data(), tensor.dataSize, tensor.data, tensor.dataSize, ASDRT_MEMCOPY_DEVICE_TO_HOST);
        ASD_LOG_IF(st != 0, ERROR) << "AsdRtMemCopy device to host fail for save tensor, ret:" << st;
        binFile.AddObject("data", hostData.data(), tensor.dataSize);
    } else {
        ASD_LOG(INFO) << "save asdtensor " << filePath << " data is empty";
    }
    AsdOps::Status st = binFile.Write(filePath);
    if (st.Ok()) {
        ASD_LOG(INFO) << "save asdtensor " << filePath;
    } else {
        ASD_LOG(ERROR) << "save asdtensor " << filePath << " fail, error:" << st.Message();
    }
}

void TensorUtil::SaveVariantPack(Handle &handle, const VariantPack &variantPack, const std::string &dirPath)
{
    AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
        std::string fileName = "inTensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        SaveTensor(variantPack.inTensors.at(i), filePath);
    }

    for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
        std::string fileName = "outTensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        SaveTensor(variantPack.outTensors.at(i), filePath);
    }
}

void TensorUtil::SaveVariantPack(Handle &handle, const RunnerVariantPack &runnerVariantPack, const std::string &dirPath)
{
    AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for (size_t i = 0; i < runnerVariantPack.inTensors.size(); ++i) {
        std::string fileName = "inTensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        SaveTensor(runnerVariantPack.inTensors.at(i), filePath);
    }

    for (size_t i = 0; i < runnerVariantPack.outTensors.size(); ++i) {
        std::string fileName = "outTensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        SaveTensor(runnerVariantPack.outTensors.at(i), filePath);
    }
}

void TensorUtil::SaveRunInfo(Handle &handle, const AsdOps::RunInfo &runInfo, const std::string &dirPath)
{
    AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for (size_t i = 0; i < runInfo.GetInTensorCount(); ++i) {
        std::string fileName = "intensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        SaveTensor(runInfo.GetInTensor(i), filePath);
    }

    for (size_t i = 0; i < runInfo.GetOutTensorCount(); ++i) {
        std::string fileName = "outtensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = AsdOps::FileSystem::Join({dirPath, fileName});
        SaveTensor(runInfo.GetOutTensor(i), filePath);
    }
}

std::string TensorUtil::AsdOpsDimsToString(const AsdOps::SVector<int64_t> &dims)
{
    std::string str;
    for (size_t i = 0; i < dims.size(); ++i) {
        str.append(std::to_string(dims.at(i)));
        if (i != dims.size() - 1) {
            str.append(",");
        }
    }
    return str;
}

int64_t TensorUtil::AlignInt(int64_t value, int align) { return (value + (align - 1)) / align * align; }

std::string TensorUtil::AsdOpsRunInfoToString(const AsdOps::RunInfo &kernelRunInfo)
{
    std::stringstream ss;
    ss << "opdesc.opName:" << kernelRunInfo.GetOpDesc().opName << ", stream:" << kernelRunInfo.GetStream() << std::endl;

    for (size_t i = 0; i < kernelRunInfo.GetInTensorCount(); ++i) {
        ss << "intensors[" << i << "]: " << TensorUtil::AsdOpsTensorToString(kernelRunInfo.GetInTensor(i)) << std::endl;
    }
    for (size_t i = 0; i < kernelRunInfo.GetOutTensorCount(); ++i) {
        ss << "outtensors[" << i << "]: " << TensorUtil::AsdOpsTensorToString(kernelRunInfo.GetOutTensor(i));
        if (i != kernelRunInfo.GetOutTensorCount() - 1) {
            ss << std::endl;
        }
    }
    return ss.str();
}
} // namespace AclTransformer