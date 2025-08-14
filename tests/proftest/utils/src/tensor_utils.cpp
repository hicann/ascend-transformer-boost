/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_utils.h"
#include "type_utils.h"
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <atb/utils/log.h>

size_t GetDataItemSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_DT_UNDEFINED:
            return sizeof(bool);
        case ACL_BOOL:
            return sizeof(bool);
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_FLOAT16:
            return sizeof(uint16_t);
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_INT16:
            return sizeof(int16_t);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        case ACL_UINT8:
            return sizeof(uint8_t);
        case ACL_UINT16:
            return sizeof(uint16_t);
        case ACL_UINT32:
            return sizeof(uint32_t);
        case ACL_UINT64:
            return sizeof(uint64_t);
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_DOUBLE:
            return sizeof(double);
        default:
            return 0;
    }
}

static std::mt19937 gen(0);

template <typename T> T random_float(float min, float max)
{
    std::uniform_real_distribution<T> dist(min, max);
    return dist(gen);
}


template <typename T> T random_int(float min, float max)
{
    int32_t min_int32 = static_cast<int32_t>(std::round(min));
    int32_t max_int32 = static_cast<int32_t>(std::round(max));
    std::uniform_int_distribution<T> dist(min_int32, max_int32);
    return dist(gen);
}

template <typename T> T random_uint(float min, float max)
{
    int32_t min_int32 = static_cast<int32_t>(std::round(min));
    int32_t max_int32 = static_cast<int32_t>(std::round(max));
    min_int32 = min_int32 < 0 ? 0 : min_int32;
    max_int32 = max_int32 < 0 ? 0 : max_int32;
    std::uniform_int_distribution<T> dist(min_int32, max_int32);
    return dist(gen);
}

bool random_bool()
{
    std::uniform_int_distribution<int> dist(0, 1);
    return dist(gen);
}

atb::Tensor FillTensorDataRandomly(const atb::TensorDesc &desc, float range_min, float range_max)
{
    atb::Tensor tensor{desc, nullptr, nullptr, 0};
    tensor.dataSize = atb::Utils::GetTensorSize(desc);
    aclrtMallocHost((void **)&tensor.hostData, tensor.dataSize);
    {
        size_t dataItemSize = GetDataItemSize(desc.dtype);
        uint64_t tensorNumel = atb::Utils::GetTensorNumel(desc);
        void *basePtr = static_cast<void *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; ++i) {
            void *elementPtr = static_cast<char *>(basePtr) + i * dataItemSize;
            switch (desc.dtype) {
                case ACL_FLOAT:
                    *static_cast<float *>(elementPtr) = random_float<float>(range_min, range_max);
                    break;
                case ACL_DOUBLE:
                    *static_cast<double *>(elementPtr) = random_float<double>(range_min, range_max);
                    break;
                case ACL_INT8:
                    *static_cast<int8_t *>(elementPtr) = random_int<int8_t>(range_min, range_max);
                    break;
                case ACL_INT16:
                    *static_cast<int16_t *>(elementPtr) = random_int<int16_t>(range_min, range_max);
                    break;
                case ACL_INT32:
                    *static_cast<int32_t *>(elementPtr) = random_int<int32_t>(range_min, range_max);
                    break;
                case ACL_INT64:
                    *static_cast<int64_t *>(elementPtr) = random_int<int64_t>(range_min, range_max);
                    break;
                case ACL_UINT8:
                    *static_cast<uint8_t *>(elementPtr) = random_uint<uint8_t>(range_min, range_max);
                    break;
                case ACL_UINT16:
                    *static_cast<uint16_t *>(elementPtr) = random_uint<uint16_t>(range_min, range_max);
                    break;
                case ACL_UINT32:
                    *static_cast<uint32_t *>(elementPtr) = random_uint<uint32_t>(range_min, range_max);
                    break;
                case ACL_UINT64:
                    *static_cast<uint64_t *>(elementPtr) = random_uint<uint64_t>(range_min, range_max);
                    break;
                case ACL_BOOL:
                    *static_cast<bool *>(elementPtr) = random_bool();
                    break;
                case ACL_FLOAT16:
                    *static_cast<uint16_t *>(elementPtr) = FloatToFloat16(random_float<float>(range_min, range_max));
                    break;
                case ACL_BF16:
                    *static_cast<uint16_t *>(elementPtr) = FloatToBfloat16(random_float<float>(range_min, range_max));
                    break;
                default:
                    break;
            }
        }
    }
    aclrtMalloc((void **)&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tensor.deviceData, tensor.dataSize, tensor.hostData, tensor.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensor;
}

atb::Tensor FillTensorDataRandomly(const atb::TensorDesc &desc)
{
    atb::Tensor tensor = FillTensorDataRandomly(desc, -5, 5);
    return tensor;
}

atb::Tensor FillTensorDataRandomly(const atb::TensorDesc &desc, const std::pair<float, float> range)
{
    atb::Tensor tensor = FillTensorDataRandomly(desc, range.first, range.second);
    return tensor;
}

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs)
{
    std::vector<atb::Tensor> tensors;
    for (const atb::TensorDesc &desc : descs) {
        atb::Tensor tensor = FillTensorDataRandomly(desc);
        tensors.push_back(tensor);
    }

    return tensors;
}

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs,
                                                const std::vector<float> &range_mins,
                                                const std::vector<float> &range_maxs)
{
    std::vector<atb::Tensor> tensors;
    if (range_mins.size() != range_maxs.size()) {
        std::cout << "range_mins.size() != range_maxs.size()" << std::endl;
        return tensors;
    }
    if (descs.size() < range_mins.size()) {
        std::cout << "descs.size() < ranges.size(), The range in the back will be discarded" << std::endl;
    } else if (descs.size() > range_mins.size()) {
        std::cout << "descs.size() > ranges.size(), The tensor in the back will be filled with zero" << std::endl;
    }

    for (size_t i = 0; i < descs.size(); ++i) {
        if (i < range_mins.size()) {
            atb::Tensor tensor = FillTensorDataRandomly(descs[i], range_mins[i], range_maxs[i]);
            tensors.push_back(tensor);
        } else {
            atb::Tensor tensor = FillTensorDataByZero(descs[i]);
            tensors.push_back(tensor);
        }
    }
    return tensors;
}

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs, float range_min,
                                                float range_max)
{
    std::vector<atb::Tensor> tensors;
    for (size_t i = 0; i < descs.size(); ++i) {
        atb::Tensor tensor = FillTensorDataRandomly(descs[i], range_min, range_max);
        tensors.push_back(tensor);
    }
    return tensors;
}

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs,
                                                const std::vector<std::pair<float, float>> &ranges)
{
    std::vector<atb::Tensor> tensors;
    if (descs.size() < ranges.size()) {
        std::cout << "descs.size() < ranges.size(), The range in the back will be discarded" << std::endl;
    } else if (descs.size() > ranges.size()) {
        std::cout << "descs.size() > ranges.size(), The tensor in the back will be filled with zero" << std::endl;
    }

    for (size_t i = 0; i < descs.size(); ++i) {
        if (i < ranges.size()) {
            atb::Tensor tensor = FillTensorDataRandomly(descs[i], ranges[i]);
            tensors.push_back(tensor);
        } else {
            atb::Tensor tensor = FillTensorDataByZero(descs[i]);
            tensors.push_back(tensor);
        }
    }
    return tensors;
}

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs,
                                                std::pair<float, float> range)
{
    std::vector<atb::Tensor> tensors;
    for (size_t i = 0; i < descs.size(); ++i) {
        atb::Tensor tensor = FillTensorDataRandomly(descs[i], range);
        tensors.push_back(tensor);
    }
    return tensors;
}

atb::Tensor FillTensorDataByZero(const atb::TensorDesc &desc)
{
    atb::Tensor tensor{desc, nullptr, nullptr, 0};
    tensor.dataSize = atb::Utils::GetTensorSize(desc);
    aclrtMallocHost((void **)&tensor.hostData, tensor.dataSize);
    {
        size_t dataItemSize = GetDataItemSize(desc.dtype);
        uint64_t tensorNumel = atb::Utils::GetTensorNumel(desc);
        void *basePtr = static_cast<void *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; ++i) {
            void *elementPtr = static_cast<char *>(basePtr) + i * dataItemSize;
            switch (desc.dtype) {
                case ACL_FLOAT:
                    *static_cast<float *>(elementPtr) = 0.0f;
                    break;
                case ACL_DOUBLE:
                    *static_cast<double *>(elementPtr) = 0.0;
                    break;
                case ACL_INT8:
                    *static_cast<int8_t *>(elementPtr) = 0;
                    break;
                case ACL_INT16:
                    *static_cast<int16_t *>(elementPtr) = 0;
                    break;
                case ACL_INT32:
                    *static_cast<int32_t *>(elementPtr) = 0;
                    break;
                case ACL_INT64:
                    *static_cast<int64_t *>(elementPtr) = 0;
                    break;
                case ACL_UINT8:
                    *static_cast<uint8_t *>(elementPtr) = 0;
                    break;
                case ACL_UINT16:
                    *static_cast<uint16_t *>(elementPtr) = 0;
                    break;
                case ACL_UINT32:
                    *static_cast<uint32_t *>(elementPtr) = 0;
                    break;
                case ACL_UINT64:
                    *static_cast<uint64_t *>(elementPtr) = 0;
                    break;
                case ACL_BOOL:
                    *static_cast<bool *>(elementPtr) = false;
                    break;
                case ACL_FLOAT16:
                    *static_cast<uint16_t *>(elementPtr) = FloatToFloat16(0.0f);
                    break;
                case ACL_BF16:
                    *static_cast<uint16_t *>(elementPtr) = FloatToBfloat16(0.0f);
                    break;
                default:
                    break;
            }
        }
    }
    aclrtMalloc((void **)&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tensor.deviceData, tensor.dataSize, tensor.hostData, tensor.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    return tensor;
}

std::vector<atb::Tensor> FillTensorDataByZero(const std::vector<atb::TensorDesc> &descs)
{
    std::vector<atb::Tensor> tensors;
    for (const atb::TensorDesc &desc : descs) {
        atb::Tensor tensor = FillTensorDataByZero(desc);
        tensors.push_back(tensor);
    }

    return tensors;
}

atb::Tensor FillTensorDataByOne(const atb::TensorDesc &desc)
{
    atb::Tensor tensor{desc, nullptr, nullptr, 0};
    tensor.dataSize = atb::Utils::GetTensorSize(desc);
    aclrtMallocHost((void **)&tensor.hostData, tensor.dataSize);
    {
        size_t dataItemSize = GetDataItemSize(desc.dtype);
        uint64_t tensorNumel = atb::Utils::GetTensorNumel(desc);
        void *basePtr = static_cast<void *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; ++i) {
            void *elementPtr = static_cast<char *>(basePtr) + i * dataItemSize;
            switch (desc.dtype) {
                case ACL_FLOAT:
                    *static_cast<float *>(elementPtr) = 1.0f;
                    break;
                case ACL_DOUBLE:
                    *static_cast<double *>(elementPtr) = 1.0;
                    break;
                case ACL_INT8:
                    *static_cast<int8_t *>(elementPtr) = 1;
                    break;
                case ACL_INT16:
                    *static_cast<int16_t *>(elementPtr) = 1;
                    break;
                case ACL_INT32:
                    *static_cast<int32_t *>(elementPtr) = 1;
                    break;
                case ACL_INT64:
                    *static_cast<int64_t *>(elementPtr) = 1;
                    break;
                case ACL_UINT8:
                    *static_cast<uint8_t *>(elementPtr) = 1;
                    break;
                case ACL_UINT16:
                    *static_cast<uint16_t *>(elementPtr) = 1;
                    break;
                case ACL_UINT32:
                    *static_cast<uint32_t *>(elementPtr) = 1;
                    break;
                case ACL_UINT64:
                    *static_cast<uint64_t *>(elementPtr) = 1;
                    break;
                case ACL_BOOL:
                    *static_cast<bool *>(elementPtr) = true;
                    break;
                case ACL_FLOAT16:
                    *static_cast<uint16_t *>(elementPtr) = FloatToFloat16(1.0f);
                    break;
                case ACL_BF16:
                    *static_cast<uint16_t *>(elementPtr) = FloatToBfloat16(1.0f);
                    break;
                default:
                    break;
            }
        }
    }
    aclrtMalloc((void **)&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tensor.deviceData, tensor.dataSize, tensor.hostData, tensor.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensor;
}

std::vector<atb::Tensor> FillTensorDataByOne(const std::vector<atb::TensorDesc> &descs)
{
    std::vector<atb::Tensor> tensors;
    for (const atb::TensorDesc &desc : descs) {
        atb::Tensor tensor = FillTensorDataByOne(desc);
        tensors.push_back(tensor);
    }

    return tensors;
}

atb::Tensor FillTensorDataByFile(const atb::TensorDesc &desc, const std::string &filePath)
{
    std::fstream file(filePath, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Can't open: " << filePath << std::endl;
        exit(1);
    }
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> fileData(fileSize);
    file.read(fileData.data(), fileSize);
    file.close();
    size_t begin_offset = 0;
    size_t data_start = 0;
    const std::string end_marker = "$End=1";
    for (size_t i = 0; i < fileSize; ++i) {
        if (fileData[i] == '\n') {
            std::string line(fileData.data() + begin_offset, fileData.data() + i);
            begin_offset = i + 1;
            if (line.find(end_marker) != std::string::npos) {
                data_start = i + 1;
                break;
            }
        }
    }

    size_t binary_size = fileSize - data_start;
    atb::Tensor tensor{desc, nullptr, nullptr, 0};
    tensor.dataSize = atb::Utils::GetTensorSize(desc);
    if (binary_size < tensor.dataSize) {
        std::cerr << "binary_size < tensor.dataSize" << "\n"
                  << "filePath:" << filePath << "\n"
                  << "binary_size: " << binary_size << " tensor.dataSize: " << tensor.dataSize << std::endl;
        exit(1);
    }
    aclrtMallocHost((void **)&tensor.hostData, tensor.dataSize);
    aclrtMemcpy(tensor.hostData, tensor.dataSize, fileData.data() + data_start, tensor.dataSize,
                ACL_MEMCPY_HOST_TO_HOST);
    aclrtMalloc((void **)&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tensor.deviceData, tensor.dataSize, tensor.hostData, tensor.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensor;
}

std::vector<atb::Tensor> FillTensorDataByFile(const std::vector<atb::TensorDesc> &descs,
                                              const std::vector<std::string> &filePaths)
{
    std::vector<atb::Tensor> tensors;
    if (descs.size() < filePaths.size()) {
        std::cout << "descs.size() < filePaths.size(), The filePath in the back will be discarded" << std::endl;
    } else if (descs.size() > filePaths.size()) {
        std::cout << "descs.size() > filePaths.size(), The tensor in the back will be filled with zero" << std::endl;
    }

    for (size_t i = 0; i < descs.size(); ++i) {
        if (i < filePaths.size()) {
            atb::Tensor tensor = FillTensorDataByFile(descs[i], filePaths[i]);
            tensors.push_back(tensor);
        } else {
            atb::Tensor tensor = FillTensorDataByZero(descs[i]);
            tensors.push_back(tensor);
        }
    }

    return tensors;
}

void FreeTensor(atb::Tensor &tensor)
{
    if (tensor.hostData != nullptr) {
        aclrtFreeHost(tensor.hostData);
        tensor.hostData = nullptr;
    }
    if (tensor.deviceData != nullptr) {
        aclrtFree(tensor.deviceData);
        tensor.deviceData = nullptr;
    }
}

void FreeTensor(std::vector<atb::Tensor> &tensors)
{
    for (atb::Tensor &tensor : tensors) {
        FreeTensor(tensor);
    }
}

std::string FormatPrintTensorData(aclDataType dtype, size_t dataItemSize, void *data, uint64_t dimNum,
                                  const int64_t *dims, size_t offset = 0, uint64_t depth = 0)
{
    if (depth == dimNum) {
        void *elementPtr = static_cast<char *>(data) + offset;
        switch (dtype) {
            case ACL_BOOL:
                return *static_cast<bool *>(elementPtr) ? "true" : "false";
            case ACL_FLOAT:
                {
                    float val = *static_cast<float *>(elementPtr);
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6) << val;
                    return oss.str();
                }
            case ACL_FLOAT16:
                {
                    float16 val = *static_cast<float16 *>(elementPtr);
                    float fval = Float16ToFloat(val);
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6) << fval;
                    return oss.str();
                }
            case ACL_INT8:
                return std::to_string(*static_cast<int8_t *>(elementPtr));
            case ACL_INT16:
                return std::to_string(*static_cast<int16_t *>(elementPtr));
            case ACL_INT32:
                return std::to_string(*static_cast<int32_t *>(elementPtr));
            case ACL_INT64:
                return std::to_string(*static_cast<int64_t *>(elementPtr));
            case ACL_UINT8:
                return std::to_string(*static_cast<uint8_t *>(elementPtr));
            case ACL_UINT16:
                return std::to_string(*static_cast<uint16_t *>(elementPtr));
            case ACL_UINT32:
                return std::to_string(*static_cast<uint32_t *>(elementPtr));
            case ACL_UINT64:
                return std::to_string(*static_cast<uint64_t *>(elementPtr));
            case ACL_BF16:
                {
                    bfloat16 val = *static_cast<bfloat16 *>(elementPtr);
                    float fval = Bfloat16ToFloat(val);
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6) << fval;
                    return oss.str();
                }
            case ACL_DOUBLE:
                {
                    double val = *static_cast<double *>(elementPtr);
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6) << val;
                    return oss.str();
                }
            default:
                return "unsupported";
        }
    }
    size_t stride = dataItemSize;
    for (size_t i = depth + 1; i < dimNum; ++i) {
        stride *= dims[i];
    }
    std::ostringstream oss;
    oss << "[";
    for (uint64_t i = 0; i < dims[depth]; ++i) {
        if (i > 0) {
            if (depth == dimNum - 1) {
                oss << ",";
            } else {
                oss << "\n";
            }
        }
        oss << FormatPrintTensorData(dtype, dataItemSize, data, dimNum, dims, offset + i * stride, depth + 1);
    }
    oss << "]";
    return oss.str();
}


std::string GetDataTypeString(aclDataType dtype)
{
    switch (dtype) {
        case ACL_DT_UNDEFINED:
            return "ACL_DT_UNDEFINED";
        case ACL_BOOL:
            return "ACL_BOOL";
        case ACL_FLOAT:
            return "ACL_FLOAT";
        case ACL_FLOAT16:
            return "ACL_FLOAT16";
        case ACL_INT8:
            return "ACL_INT8";
        case ACL_INT16:
            return "ACL_INT16";
        case ACL_INT32:
            return "ACL_INT32";
        case ACL_INT64:
            return "ACL_INT64";
        case ACL_UINT8:
            return "ACL_UINT8";
        case ACL_UINT16:
            return "ACL_UINT16";
        case ACL_UINT32:
            return "ACL_UINT32";
        case ACL_UINT64:
            return "ACL_UINT64";
        case ACL_BF16:
            return "ACL_BF16";
        case ACL_DOUBLE:
            return "ACL_DOUBLE";
        default:
            return "";
    }
}
std::string FormatPrintTensorDesc(const atb::Tensor &tensor)
{
    std::ostringstream oss;
    oss << "aclDataType: " << GetDataTypeString(tensor.desc.dtype) << ", dim: [";
    for (size_t i = 0; i < tensor.desc.shape.dimNum; ++i) {
        oss << std::to_string(tensor.desc.shape.dims[i]);
        if (i != tensor.desc.shape.dimNum - 1) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

void PrintDeviceTensor(const atb::Tensor &tensor)
{
    if (tensor.deviceData == nullptr) {
        std::cout << "tensor's daviceData == nullptr, no print\n";
        return;
    }
    void *hostData;
    aclrtMallocHost((void **)&hostData, tensor.dataSize);
    aclrtMemcpy(hostData, tensor.dataSize, tensor.deviceData, tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    {
        size_t dataItemSize = GetDataItemSize(tensor.desc.dtype);
        std::cout << FormatPrintTensorData(tensor.desc.dtype, dataItemSize, hostData, tensor.desc.shape.dimNum,
                                           tensor.desc.shape.dims)
                  << " " << FormatPrintTensorDesc(tensor) << std::endl;
    }
    aclrtFreeHost(hostData);
}


void PrintDeviceTensor(const std::vector<atb::Tensor> &tensors)
{
    for (const atb::Tensor &tensor : tensors) {
        PrintDeviceTensor(tensor);
    }
}
