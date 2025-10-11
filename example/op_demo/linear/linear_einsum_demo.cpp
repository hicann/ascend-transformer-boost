/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../demo_util.h"

const int32_t DEVICE_ID = 0;
const uint32_t XDIM_0 = 32;
const uint32_t XDIM_1 = 512;
const uint32_t WEIGHTDIM_0 = 512;
const uint32_t WEIGHTDIM_1 = 128;
const uint32_t BATCHSIZE = 128;

/**
 * @brief 准备atb::VariantPack
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建shape为[32, 128, 512]的输入x tensor
    atb::Tensor xFloat;
    CreateTensorFromVector(contextPtr, stream, std::vector<float>(XDIM_0 * BATCHSIZE * XDIM_1, 1),
                           aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {XDIM_0, BATCHSIZE, XDIM_1}, xFloat);
    // 创建shape为[128, 512, 128]的输入weight tensor
    atb::Tensor weightFloat;
    CreateTensorFromVector(contextPtr, stream, std::vector<float>(BATCHSIZE * WEIGHTDIM_0 * WEIGHTDIM_1, 2),
                           aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {BATCHSIZE, WEIGHTDIM_0, WEIGHTDIM_1},
                           weightFloat);
    inTensors = {xFloat, weightFloat};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个linear operation
 * @param linearOp 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status CreateLinearOperation(atb::Operation **linearOp)
{
    atb::infer::LinearParam param;
    param.transposeA = false;
    param.transposeB = false;
    param.hasBias = false;
    param.outDataType = aclDataType::ACL_DT_UNDEFINED;
    param.enAccum = false;
    param.matmulType = atb::infer::LinearParam::MatmulType::MATMUL_EIN_SUM;
    param.quantMode = atb::infer::LinearParam::QuantMode::QUANT_UNDEFINED;
    return atb::CreateOperation(param, linearOp);
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
    atb::Operation *linearOp = nullptr;
    CHECK_STATUS(CreateLinearOperation(&linearOp));
    // 准备输入tensor
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareInTensor(context, stream, variantPack.inTensors)); // 放入输入tensor
    // 准备输出tensor
    atb::Tensor output;
    CHECK_STATUS(
        CreateTensor(aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {XDIM_0, BATCHSIZE, WEIGHTDIM_1}, output));
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
    std::cout << "Linear EinSum demo success!" << std::endl;
    return 0;
}
