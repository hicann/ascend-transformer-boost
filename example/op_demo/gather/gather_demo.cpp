/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../demo_util.h"

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param seqLenHost host侧tensor。序列长度向量，等于1时，为增量或全量；大于1时，为全量
 * @param tokenOffsetHost host侧tensor。计算完成后的token偏移
 * @param layerId layerId，取cache的kv中哪一个kv进行计算
 * @return atb::SVector<atb::Tensor> atb::VariantPack中的输入tensor
 * @note 需要传入所有host侧tensor
 */
atb::SVector<atb::Tensor> PrepareInTensor(atb::Context *contextPtr, aclrtStream stream)
{
    uint32_t dim0 = 3;
    uint32_t dim1 = 3;
    // 创建tensor0
    std::vector<float> tensorzero{1, 2, 3, 4, 5, 6, 7, 8, 9};
    atb::Tensor tensorZero =
        CreateTensorFromVector(contextPtr, stream, tensorzero, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {dim0, dim1});
    // 创建tensor1
    std::vector<int64_t> tensorone{2, 1};
    atb::Tensor tensorOne =
        CreateTensorFromVector(contextPtr, stream, tensorone, ACL_INT64, aclFormat::ACL_FORMAT_ND, {2});
    // 根据顺序将所有输入tensor放入SVector
    atb::SVector<atb::Tensor> inTensors = {tensorZero, tensorOne};
    return inTensors;
}

/**
 * @brief 创建一个Gather的Operation，并设置参数
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *PrepareOperation()
{
    atb::infer::GatherParam gatherParam;
    gatherParam.axis = 0;
    gatherParam.batchDims = 0;
    atb::Operation *op = nullptr;
    CHECK_STATUS(atb::CreateOperation(gatherParam, &op));
    return op;
}

int main(int argc, char **argv)
{
    // 1.设置卡号、创建context、设置stream
    CHECK_STATUS(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // Gather示例
    atb::Operation *op = PrepareOperation();
    // 准备输入张量
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareInTensor(context, stream);                            // 放入输入tensor
    atb::Tensor tensorOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {2, 3}); // 创建输出tensor
    variantPack.outTensors.push_back(tensorOut);                                         // 放入输出tensor

    // setup阶段，计算workspace大小
    uint64_t workspaceSize = 0;
    CHECK_STATUS(op->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // execute阶段
    op->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    // 释放内存
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(op)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context)); // context，全局资源，后释放
    std::cout << "Gather demo success!" << std::endl;
    return 0;
}
