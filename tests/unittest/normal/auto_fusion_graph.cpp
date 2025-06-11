/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
// #include <torch/torch.h>
#include <atb/utils/log.h>
#include <atb/utils.h>
#include "test_utils/test_common.h"
#include "atb/operation.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "atb/utils/operation_util.h"
#include "atb/operation_infra.h"
#include "atb/operation/operation_base.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/auto_fusion.h"

using namespace atb;

static void CreateMatMulAddGraphOperation(atb::GraphParam &opGraph, atb::Operation **operation,
    bool autoFusion = false)
{
	// 构子图流程
    opGraph.inTensorNum = 6;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 4;
    opGraph.nodes.resize(5);

    size_t nodeId = 0;
    atb::Node &linNode1 = opGraph.nodes.at(nodeId++);
    atb::Node &addNode1 = opGraph.nodes.at(nodeId++);
    atb::Node &linNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam, &addNode.operation);
    addNode.inTensorIds = {0, 1};
    addNode.outTensorIds = {3};

    atb::infer::LinearParam linearParam1;
    linearParam1.hasBias = false;
    linearParam1.transposeB = true;
    atb::CreateOperation(linearParam1, &linNode1.operation);
    linNode1.inTensorIds = {0, 1};
    linNode1.outTensorIds = {6};

    atb::infer::ElewiseParam addParam1;
    addParam1.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam1, &addNode1.operation);
    addNode1.inTensorIds = {2, 6};
    addNode1.outTensorIds = {7};


    atb::infer::LinearParam linearParam2;
    linearParam2.hasBias = false;
    linearParam2.transposeB = true;
    atb::CreateOperation(linearParam2, &linNode2.operation);
    linNode2.inTensorIds = {3, 4};
    linNode2.outTensorIds = {8};

    atb::infer::ElewiseParam addParam2;
    addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam2, &addNode2.operation);
    addNode2.inTensorIds = {5, 8};
    addNode2.outTensorIds = {9};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &addNode.operation);
    addNode.inTensorIds = {7, 9};
    addNode.outTensorIds = {10};
    if (true == autoFusion) {
        atb::AutoFusion autoFusiontool(opGraph);
        autoFusiontool.DoAutoFusion();
    }

    atb::CreateOperation(opGraph, operation);
}

static void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs)
{
    for (size_t i = 0; i < intensorDescs.size(); i++) {
        intensorDescs.at(i).dtype = ACL_BF16;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = 2;
        intensorDescs.at(i).shape.dims[0] = 2;
        intensorDescs.at(i).shape.dims[1] = 2;
    }
}

static void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
{
    std::vector<char> zeroData(24, 1); // 一段全0的hostBuffer
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        int ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
        ret = aclrtMemcpy(inTensors.at(i).deviceData, inTensors.at(i).dataSize, zeroData.data(), zeroData.size(), ACL_MEMCPY_HOST_TO_DEVICE); //拷贝CPU内存到NPU侧
    }
}

static void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++) {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
}


TEST(CreateMatMulAddGraphOperation, TestMatMmulAdd)
{
    uint32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    atb::Context *context = nullptr;
    Status st = atb::CreateContext(&context);
    ATB_LOG_IF(st != 0, ERROR) << "CreateContext fail";
    
    aclrtStream stream = nullptr;
    st = aclrtCreateStream(&stream);
    ATB_LOG_IF(st != 0, ERROR) << "aclrtCreateStream fail";
    context->SetExecuteStream(stream);

    atb::Operation *graphOp = nullptr;
    atb::GraphParam graphParam1;
    CreateMatMulAddGraphOperation(graphParam1, &graphOp);

    // 准备输入输出tensor
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs1;
    atb::SVector<atb::TensorDesc> outtensorDescs1;

    uint32_t inTensorNum = 6;
    uint32_t outTensorNum = 1;
    pack.inTensors.resize(inTensorNum);
    pack.outTensors.resize(outTensorNum);
    intensorDescs1.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs1);
    CreateInTensors(pack.inTensors, intensorDescs1);

    outtensorDescs1.resize(outTensorNum);
    outtensorDescs1.at(0).dtype = ACL_BF16;
    outtensorDescs1.at(0).format = ACL_FORMAT_ND;
    outtensorDescs1.at(0).shape.dimNum = 2;
    outtensorDescs1.at(0).shape.dims[0] = 2;
    outtensorDescs1.at(0).shape.dims[1] = 2;

    // outtensorDescs1.at(1).dtype = ACL_FLOAT16;
    // outtensorDescs1.at(1).format = ACL_FORMAT_ND;
    // outtensorDescs1.at(1).shape.dimNum = 2;
    // outtensorDescs1.at(1).shape.dims[0] = 2;
    // outtensorDescs1.at(1).shape.dims[1] = 2;
    // pack.outTensors.resize(outTensorNum);
    CreateOutTensors(pack.outTensors, outtensorDescs1);

    // Setup
    uint64_t workspaceSize = 0;
    graphOp->Setup(pack, workspaceSize, context);
    void *workSpace = nullptr;
    int ret1 = 0;
    if (workspaceSize != 0) {
        ret1 = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret1 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    //Execute
    atb::Status st1 = graphOp->Execute(pack, (uint8_t *)workSpace, workspaceSize, context);

    st = (st1 == atb::NO_ERROR) ? atb::NO_ERROR : atb::ERROR_INVALID_GRAPH;

    //流同步
    ret1 = aclrtSynchronizeStream(stream);
    EXPECT_EQ(ret1, atb::NO_ERROR);
    if (ret1 != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    // 资源释放
    atb::DestroyContext(context);
    atb::DestroyOperation(graphOp);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    ASSERT_EQ(st, atb::NO_ERROR);
    // aclFinalize();
}
