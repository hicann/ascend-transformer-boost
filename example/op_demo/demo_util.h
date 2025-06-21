/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DEMO_UTIL_H
#define DEMO_UTIL_H

#include <iostream>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "atb/atb_infer.h"
#include "atb/operation.h"
#include "atb/types.h"

#define CHECK_STATUS(status)                                                                                           \
    do {                                                                                                               \
        if ((status) != 0) {                                                                                           \
            std::cout << __FILE__ << ":" << __LINE__ << " [error]: " << (status) << std::endl;                         \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_STATUS_EXPR(status, expr)                                                                                \
    do {                                                                                                               \
        if ((status) != 0) {                                                                                           \
            std::cout << __FILE__ << ":" << __LINE__ << " [error]: " << (status) << std::endl;                         \
            expr;                                                                                                      \
        }                                                                                                              \
    } while (0)

/**
 * @brief 创建一个Tensor对象
 * @param  dataType 数据类型
 * @param  format 数据格式
 * @param  shape 数据shape
 * @return atb::Tensor 返回创建的Tensor对象
 */
atb::Tensor CreateTensor(const aclDataType dataType, const aclFormat format, std::vector<int64_t> shape)
{
    atb::Tensor tensor;
    tensor.desc.dtype = dataType;
    tensor.desc.format = format;
    tensor.desc.shape.dimNum = shape.size();
    // tensor的dim依次设置为shape中元素
    for (size_t i = 0; i < shape.size(); i++) {
        tensor.desc.shape.dims[i] = shape.at(i);
    }
    tensor.dataSize = atb::Utils::GetTensorSize(tensor); // 计算Tensor的数据大小
    CHECK_STATUS(aclrtMalloc(&tensor.deviceData, tensor.dataSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
    return tensor;
}

/**
 * @brief 进行数据类型转换，调用Elewise的cast Op
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensor 输入tensor
 * @param outTensorType 输出tensor的数据类型
 * @param shape 输出tensor的shape
 * @return atb::Tensor 转换后的tensor
 */
atb::Tensor CastOp(atb::Context *contextPtr, aclrtStream stream, const atb::Tensor inTensor,
                   const aclDataType outTensorType, std::vector<int64_t> shape)
{
    uint64_t workspaceSize = 0;
    void *workspace = nullptr;
    // 创建Elewise的ELEWISE_CAST
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_CAST;
    castParam.outTensorType = outTensorType;
    atb::Operation *castOp = nullptr;
    CHECK_STATUS(CreateOperation(castParam, &castOp));
    atb::Tensor outTensor = CreateTensor(outTensorType, aclFormat::ACL_FORMAT_ND, shape); // cast输出tensor
    atb::VariantPack castVariantPack;                                                     // 参数包
    castVariantPack.inTensors = {inTensor};
    castVariantPack.outTensors = {outTensor};
    // 在Setup接口调用时对输入tensor和输出tensor进行校验。
    CHECK_STATUS(castOp->Setup(castVariantPack, workspaceSize, contextPtr));
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc(&workspace, workspaceSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // ELEWISE_CAST执行
    CHECK_STATUS(castOp->Execute(castVariantPack, (uint8_t *)workspace, workspaceSize, contextPtr));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspace)); // 清理工作空间
    }
    return outTensor;
}

/**
 * @brief 简单封装，拷贝vector data中数据以创建tensor
 * @details 用于创建outTensorType类型的tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param data 输入vector数据
 * @param outTensorType 期望输出tensor数据类型
 * @param format 输出tensor的格式，即NZ，ND等
 * @param shape 输出tensor的shape
 * @return atb::Tensor 返回创建的tensor
 */
template <typename T>
atb::Tensor CreateTensorFromVector(atb::Context *contextPtr, aclrtStream stream, std::vector<T> data,
                                   const aclDataType outTensorType, const aclFormat format, std::vector<int64_t> shape,
                                   const aclDataType inTensorType = ACL_DT_UNDEFINED)
{
    atb::Tensor tensor;
    aclDataType intermediateType;
    switch (outTensorType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_DOUBLE:
            intermediateType = aclDataType::ACL_FLOAT;
            break;
        default:
            intermediateType = outTensorType;
    }
    if (inTensorType == outTensorType && inTensorType != ACL_DT_UNDEFINED) {
        intermediateType = outTensorType;
    }
    tensor = CreateTensor(intermediateType, format, shape);
    CHECK_STATUS(aclrtMemcpy(tensor.deviceData, tensor.dataSize, data.data(), sizeof(T) * data.size(),
                             ACL_MEMCPY_HOST_TO_DEVICE));
    if (intermediateType == outTensorType) {
        // 原始创建的tensor类型，不需要转换
        return tensor;
    }
    return CastOp(contextPtr, stream, tensor, outTensorType, shape);
}

// 判断soc型号是否为Atlas A2/A3
bool Is910B()
{
    const char *socName = aclrtGetSocName();
    if (!socName) {
        std::cout << "aclrtGetSocName failed!";
        return false;
    }

    const uint32_t LEN_OF_ASCEND_910B = 10;
    std::cout << "SocVersion: " << std::string(socName);
    return (std::string(socName).find("Ascend910B") != std::string::npos &&
            std::string(socName).length() > LEN_OF_ASCEND_910B) ||
           std::string(socName).find("Ascend910_93") != std::string::npos;
}

// 判断soc型号是否为Atlas推理系列
bool Is310P()
{
    const char *socName = aclrtGetSocName();
    if (!socName) {
        std::cout << "aclrtGetSocName failed!";
        return false;
    }

    const uint32_t LEN_OF_ASCEND_910B = 10;
    std::cout << "SocVersion: " << std::string(socName);
    return std::string(socName).find("Ascend310B") != std::string::npos;
}

#endif
