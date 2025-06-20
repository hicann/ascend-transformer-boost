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
const uint32_t DIM_0 = 128;
const uint32_t DIM_1 = 256;
const int32_t BEGIN_NORM_AXIS = 1;

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @return atb::SVector<atb::Tensor> atb::VariantPack中的输入tensor
 * @note 需要传入所有host侧tensor
 */
atb::SVector<atb::Tensor> PrepareInTensor(atb::Context *contextPtr, aclrtStream stream)
{
    atb::Tensor x = CreateTensorFromVector(contextPtr,
        stream,
        std::vector<float>(DIM_0 * DIM_1, 2.0),
        ACL_FLOAT16,
        aclFormat::ACL_FORMAT_ND,
        {DIM_0, DIM_1});
    atb::Tensor gamma = CreateTensorFromVector(
        contextPtr, stream, std::vector<float>(DIM_1, 2.0), ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {DIM_1});
    atb::Tensor beta = CreateTensorFromVector(
        contextPtr, stream, std::vector<float>(DIM_1, 1.0), ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {DIM_1});
    atb::SVector<atb::Tensor> inTensors = {x, gamma, beta};
    return inTensors;
}

/**
 * @brief 创建一个beginNormAxis为1的layernorm operation
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *CreateLayerNormOperation()
{
    atb::infer::LayerNormParam param;
    param.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    param.normParam.beginNormAxis = BEGIN_NORM_AXIS;
    atb::Operation *layerNormOp = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &layerNormOp));
    return layerNormOp;
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
    atb::Operation *layerNormOp = CreateLayerNormOperation();
    // 准备输入tensor
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareInTensor(context, stream);  // 放入输入tensor
    atb::Tensor tensorOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {DIM_0, DIM_1});
    variantPack.outTensors = {tensorOut};  // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspace大小
    CHECK_STATUS(layerNormOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // layernorm执行
    layerNormOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream));  // 流同步，等待device侧任务计算完成

    // 释放资源
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    CHECK_STATUS(aclrtFree(tensorOut.deviceData));
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(layerNormOp));  // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context));  // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "Layernorm demo success!" << std::endl;
    return 0;
}
