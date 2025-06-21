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

const int32_t DEVICE_ID = 0;
const uint32_t XDIM_0 = 2;
const uint32_t XDIM_1 = 3;
const uint32_t WEIGHTDIM_0 = 3;
const uint32_t WEIGHTDIM_1 = 2;

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @return atb::SVector<atb::Tensor> atb::VariantPack中的输入tensor
 * @note 需要传入所有host侧tensor
 */
atb::SVector<atb::Tensor> PrepareInTensor(atb::Context *contextPtr, aclrtStream stream)
{
    // 创建shape为[2, 3]的输入x tensor
    atb::Tensor x = CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>{1, 2, 3, 4, 5, 6},
                                           aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, {XDIM_0, XDIM_1});
    // 创建shape为[3, 2]的输入weight tensor
    atb::Tensor weight =
        CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>{1, 2, 3, 4, 5, 6}, aclDataType::ACL_INT8,
                               aclFormat::ACL_FORMAT_ND, {WEIGHTDIM_0, WEIGHTDIM_1});
    // 创建shape为[2]bias tensor
    atb::Tensor bias = CreateTensorFromVector(contextPtr, stream, std::vector<int32_t>(2, 1), aclDataType::ACL_INT32,
                                              aclFormat::ACL_FORMAT_ND, {1, 2});
    // 创建shape为[2]的输入deqScale tensor
    atb::Tensor deqScale = CreateTensorFromVector(contextPtr, stream, std::vector<float>(2, 1), aclDataType::ACL_FLOAT,
                                                  aclFormat::ACL_FORMAT_ND, {1, 2});
    atb::SVector<atb::Tensor> inTensors = {x, weight, bias, deqScale};
    return inTensors;
}

/**
 * @brief 创建一个linear operation
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *CreateLinearOperation()
{
    atb::infer::LinearParam param;
    param.transposeA = false;
    param.transposeB = false;
    param.hasBias = true;
    param.outDataType = aclDataType::ACL_BF16;
    param.enAccum = false;
    param.matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED;
    param.quantMode = PER_CHANNEL;
    atb::Operation *LinearOp = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &LinearOp));
    return LinearOp;
}

int main(int argc, char **argv)
{
    // 设置卡号、创建context、设置stream
    atb::Context *context = nullptr;
    void *stream = nullptr;

    CHECK_STATUS(aclInit(nullptr));
    if (!Is910B()) {
        std::cout << "This linear demo only supports A2/A3" << std::endl;
        CHECK_STATUS(aclFinalize());
        return 0;
    }
    
    CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
    CHECK_STATUS(atb::CreateContext(&context));
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // 创建op
    atb::Operation *linearOp = CreateLinearOperation();
    // 准备输入tensor
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareInTensor(context, stream); // 放入输入tensor
    // 准备输出tensor
    atb::Tensor output = CreateTensor(aclDataType::ACL_BF16, aclFormat::ACL_FORMAT_ND, {XDIM_0, WEIGHTDIM_1});
    variantPack.outTensors = {output}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(linearOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // linear执行
    linearOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

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
    CHECK_STATUS(atb::DestroyOperation(linearOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "Linear dequant demo for A2/A3 success!" << std::endl;
    return 0;
}