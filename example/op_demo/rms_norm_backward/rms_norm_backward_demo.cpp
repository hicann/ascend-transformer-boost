/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "demo_util.h"

const int32_t DEVICE_ID = 0;
const uint32_t DIM_0 = 32;
const uint32_t DIM_1 = 64;
const uint32_t DIM_2 = 128;

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @return atb::SVector<atb::Tensor> atb::VariantPack中的输入tensor
 * @note 需要传入所有host侧tensor
 */
atb::SVector<atb::Tensor> PrepareInTensor(atb::Context *contextPtr, aclrtStream stream)
{
    // 创建shape为[32, 64, 128]的tensor
    atb::Tensor dy = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>(DIM_0 * DIM_1 * DIM_2, 2.0),
        ACL_FLOAT,
        aclFormat::ACL_FORMAT_ND,
        {DIM_0, DIM_1, DIM_2});
    atb::Tensor x = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>(DIM_0 * DIM_1 * DIM_2, 2.0),
        ACL_FLOAT,
        aclFormat::ACL_FORMAT_ND,
        {DIM_0, DIM_1, DIM_2});
    atb::Tensor rstd = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>(DIM_0 * DIM_1, 2.0),
        ACL_FLOAT,
        aclFormat::ACL_FORMAT_ND,
        {DIM_0, DIM_1, 1});
    atb::Tensor gamma = CreateTensorFromVector(
        contextPtr, stream, std::vector<float>(DIM_2, 2.0), ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {DIM_2});
    atb::SVector<atb::Tensor> inTensors = {dy, x, rstd, gamma};
    return inTensors;
}

/**
 * @brief 创建一个rmsnorm backward operation
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *CreateRmsNormBackwardOperation()
{
    atb::train::RmsNormBackwardParam param;
    atb::Operation *rmsNormBackwardOp = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &rmsNormBackwardOp));
    return rmsNormBackwardOp;
}

int main(int argc, char **argv)
{
    // 设置卡号、创建context、设置stream
    atb::Context *context = nullptr;
    void *stream = nullptr;

    CHECK_STATUS(aclInit(nullptr));
    CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
    CHECK_STATUS(atb::CreateContext(&context));
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // 创建op
    atb::Operation *rmsnormBackwardOp = CreateRmsNormBackwardOperation();
    // 准备输入tensor
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareInTensor(context, stream);  // 放入输入tensor
    atb::Tensor dx = CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {DIM_0, DIM_1, DIM_2});
    atb::Tensor dgamma = CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {DIM_2});
    variantPack.outTensors = {dx, dgamma};  // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspace大小
    CHECK_STATUS(rmsnormBackwardOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // rmsnorm执行
    rmsnormBackwardOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream));  // 流同步，等待device侧任务计算完成

    // 释放资源
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    for (atb::Tensor &outTensor : variantPack.outTensors) {
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(rmsnormBackwardOp));  // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context));  // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "Rmsnorm backward demo success!" << std::endl;
    return 0;
}
