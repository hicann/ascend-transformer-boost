/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "atb/atb_infer.h"
#include "atb/runner/graph_runner.h"
#include "test_utils/operation_test.h"
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <string.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include "atb/utils/tensor_util.h"
#include <atb/utils/log.h>
#include "atb/utils/probe.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cpp-stub/src/stub.h>
#include "atb/utils/singleton.h"
#include "atb/utils/config.h"

using namespace atb;
using namespace Mki;

atb::Status CreateTensor(const aclDataType dataType, const aclFormat format, std::vector<int64_t> shape,
                         atb::Tensor &tensor)
{
    tensor.desc.dtype = dataType;
    tensor.desc.format = format;
    tensor.desc.shape.dimNum = shape.size();
    // tensor的dim依次设置为shape中元素
    for (size_t i = 0; i < shape.size(); i++) {
        tensor.desc.shape.dims[i] = shape.at(i);
    }
    tensor.dataSize = atb::Utils::GetTensorSize(tensor); // 计算Tensor的数据大小
    CHECK_STATUS(aclrtMalloc(&tensor.deviceData, tensor.dataSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
    return atb::ErrorType::NO_ERROR;
}

atb::Status CastOp(atb::Context *contextPtr, aclrtStream stream, const atb::Tensor inTensor,
                   const aclDataType outTensorType, atb::Tensor &outTensor)
{
    uint64_t workspaceSize = 0;
    void *workspace = nullptr;
    // 创建Elewise的ELEWISE_CAST
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_CAST;
    castParam.outTensorType = outTensorType;
    atb::Operation *castOp = nullptr;
    CHECK_STATUS(CreateOperation(castParam, &castOp));
    // atb::Tensor outTensor;
    // CreateTensor(outTensorType, aclFormat::ACL_FORMAT_ND, shape, outTensor); // cast输出tensor
    atb::VariantPack castVariantPack; // 参数包
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
    return atb::ErrorType::NO_ERROR;
}

template <typename T>
atb::Status CreateTensorFromVector(atb::Context *contextPtr, aclrtStream stream, std::vector<T> data,
                                   const aclDataType outTensorType, const aclFormat format, std::vector<int64_t> shape,
                                   atb::Tensor &outTensor, const aclDataType inTensorType = ACL_DT_UNDEFINED)
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
    CHECK_STATUS(CreateTensor(intermediateType, format, shape, tensor));
    CHECK_STATUS(aclrtMemcpy(tensor.deviceData, tensor.dataSize, data.data(), sizeof(T) * data.size(),
                             ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, shape, outTensor));
    if (intermediateType == outTensorType) {
        // 原始创建的tensor类型，不需要转换
        outTensor = tensor;
        return atb::ErrorType::NO_ERROR;
    }
    return CastOp(contextPtr, stream, tensor, outTensorType, outTensor);
}

atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    uint32_t dim0 = 2;
    uint32_t dim1 = 2;
    // 创建tensor0
    std::vector<float> tensor0(VECTOR_SIZE, INIT_VALUE);
    atb::Tensor atbTensor0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, tensor0, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {dim0, dim1}, atbTensor0));
    // 创建tensor1
    std::vector<float> tensor1(VECTOR_SIZE, INIT_VALUE);
    atb::Tensor atbTensor1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, tensor1, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {dim0, dim1}, atbTensor1));
    // 根据顺序将所有输入tensor放入SVector
    inTensors = {atbTensor0, atbTensor1};
    return atb::ErrorType::NO_ERROR;
}

bool CondFunction(void *condition)
{
    if (condition != nullptr) {
        int *data = static_cast<int *>(condition);
        return (*data > 10);
    }
    return false;
}

TEST(TestIfhOperation, IfOpTest)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        GTEST_SKIP() << "This test case only support 910B";
    }
    aclInit(nullptr);
    uint32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    std::vector<aclrtStream> streams = {stream};
    context->SetExecuteStreams(streams);

    atb::Operation *operationA;
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &operationA);
    EXPECT_EQ(status1, 0);

    atb::Operation *operationB;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &operationB);
    EXPECT_EQ(status2, 0);

    atb::Operation *ifOperation;
    atb::IfCondParam opCond;
    std::unique_ptr<int> data = std::make_unique<int>(15);
    opCond.handle = CondFunction;
    opCond.condition = data.get();
    opCond.opA = operationA;
    opCond.opB = operationB;
    atb::Status status3 = CreateOperation(opCond, &ifOperation);
    EXPECT_EQ(status3, 0);

    atb::VariantPack variantPack;
    PrepareInTensor(context, stream, variantPack.inTensors); // 放入输入tensor
    atb::Tensor tensorOut;
    CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {2, 2}, tensorOut); // 创建输出tensor
    variantPack.outTensors.push_back(tensorOut);                            // 放入输出tensor

    uint64_t workspaceSize = 0;
    atb::Status status4 = ifOperation->Setup(variantPack, workspaceSize, context);
    EXPECT_EQ(status4, 0);
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        atb::Status status5 = aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(status5, 0);
    }

    ifOperation->Execute(variantPack, workspacePtr, workspaceSize, context);
    atb::Status status6 = aclrtSynchronizeStream(stream); // 流同步，等待device侧任务计算完成
    EXPECT_EQ(status6, 0);

    for (atb::Tensor &inTensor : variantPack.inTensors) {
        atb::Status status7 = aclrtFree(inTensor.deviceData);
        EXPECT_EQ(status7, 0);
    }
    if (workspaceSize > 0) {
        atb::Status status8 = aclrtFree(workspacePtr);
        EXPECT_EQ(status8, 0);
    }
    atb::DestroyOperation(ifOperation);
    atb::DestroyContext(context);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

TEST(TestIfhOperation, IfGraphOpTest)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        GTEST_SKIP() << "This test case only support 910B";
    }

    enum InTensorId {
    IN_TENSOR_A = 0,
    IN_TENSOR_B,
    LAYER_OUT,
    ADD_OUT
    };

    aclInit(nullptr);
    uint32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    std::vector<aclrtStream> streams = {stream};
    context->SetExecuteStreams(streams);

    // graph with add+sin
    atb::Operation *operationA;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &sinNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status1 = atb::CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status1, 0);
    addNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    addNode.outTensorIds = {ADD_OUT};

    atb::infer::ElewiseParam sinParam;
    sinParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_SIN;
    atb::Status status2 = CreateOperation(sinParam, &sinNode.operation);
    EXPECT_EQ(status2, 0);
    sinNode.inTensorIds = {ADD_OUT};
    sinNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operationA);
    EXPECT_EQ(status3, 0);

    atb::Operation *operationB;
    // reuse addParam defined previously
    atb::Status status4 = CreateOperation(addParam, &operationB);
    EXPECT_EQ(status4, 0);

    atb::Operation *ifOperation;
    atb::IfCondParam opCond;
    std::unique_ptr<int> data = std::make_unique<int>(15);
    opCond.handle = CondFunction;
    opCond.condition = data.get();
    opCond.opA = operationA;
    opCond.opB = operationB;
    atb::Status status5 = CreateOperation(opCond, &ifOperation);
    EXPECT_EQ(status5, 0);

    atb::VariantPack variantPack;
    PrepareInTensor(context, stream, variantPack.inTensors); // 放入输入tensor
    atb::Tensor tensorOut;
    CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {2, 2}, tensorOut); // 创建输出tensor
    variantPack.outTensors.push_back(tensorOut);                            // 放入输出tensor

    uint64_t workspaceSize = 0;
    atb::Status status6 = ifOperation->Setup(variantPack, workspaceSize, context);
    EXPECT_EQ(status6, 0);
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        atb::Status status7 = aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(status7, 0);
    }

    ifOperation->Execute(variantPack, workspacePtr, workspaceSize, context);
    atb::Status status8 = aclrtSynchronizeStream(stream); // 流同步，等待device侧任务计算完成
    EXPECT_EQ(status8, 0);

    for (atb::Tensor &inTensor : variantPack.inTensors) {
        atb::Status status9 = aclrtFree(inTensor.deviceData);
        EXPECT_EQ(status9, 0);
    }
    if (workspaceSize > 0) {
        atb::Status status10 = aclrtFree(workspacePtr);
        EXPECT_EQ(status10, 0);
    }
    atb::DestroyOperation(ifOperation);
    atb::DestroyContext(context);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}