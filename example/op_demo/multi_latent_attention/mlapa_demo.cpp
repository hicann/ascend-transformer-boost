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

const int32_t DEVICE_ID = 1;
const uint32_t blockSize = 128;
int32_t blockNum = 64;
std::vector<int32_t> contextLensHost;
/**
 * @brief 准备atb::VariantPack
 * @param contextPtr context指针
 * @param stream stream
 * @return atb::SVector<atb::Tensor> atb::VariantPack
 * @note 需要传入所有host侧tensor
 */
atb::SVector<atb::Tensor> PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, aclDataType dtype, int tokenNum,
                                          int headNum, int kSeqLen)
{
    // 创建shape为[tokenNum, headNum, 512]的输入qNope tensor
    atb::Tensor qNope = CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(tokenNum * headNum * 512, 1),
                                               ACL_INT8, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 512});
    // 创建shape为[tokenNum, headNum, 64]的输入qRope tensor
    atb::Tensor qRope = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * headNum * 64, 0),
                                               dtype, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 64});
    int maxBlockNumPerSeq = (kSeqLen + blockSize - 1) / blockSize;
    blockNum = tokenNum * maxBlockNumPerSeq;
    // 创建shape为[blockNum, 16, blockSize, 32]的输入ctKV tensor
    atb::Tensor ctKV =
        CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(blockNum * blockSize * 512, 1), ACL_INT8,
                               aclFormat::ACL_FORMAT_FRACTAL_NZ, {blockNum, 16, blockSize, 32});
    // 创建shape为[blockNum, 4, blockSize, 16]的输入kRope tensor
    atb::Tensor kRope = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(blockNum * blockSize * 64, 0),
                                               dtype, aclFormat::ACL_FORMAT_FRACTAL_NZ, {blockNum, 4, blockSize, 16});
    // 创建shape为[tokenNum, maxBlockNumPerSeq]的输入blockTables tensor
    auto blockTablesHost = std::vector<int32_t>(tokenNum * maxBlockNumPerSeq);
    for (size_t i = 0; i < tokenNum; i++) {
        for (size_t j = 0; j < maxBlockNumPerSeq; j++) {
            blockTablesHost[i * maxBlockNumPerSeq + j] = i * maxBlockNumPerSeq + j;
        }
    }
    atb::Tensor blockTables = CreateTensorFromVector(contextPtr, stream, blockTablesHost, ACL_INT32,
                                                     aclFormat::ACL_FORMAT_ND, {tokenNum, maxBlockNumPerSeq});
    // 创建shape为[tokenNum]的输入contextLens hostTensor
    contextLensHost = std::vector<int32_t>(tokenNum, kSeqLen);
    atb::Tensor contextLens = CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {tokenNum});
    contextLens.hostData = contextLensHost.data();
    // 创建shape为[headNum]的输入qkDescale tensor
    atb::Tensor qkDescale = CreateTensorFromVector(contextPtr, stream, std::vector<float>(headNum, 0), ACL_FLOAT,
                                                   aclFormat::ACL_FORMAT_ND, {headNum});
    // 创建shape为[headNum]的输入pvDescale tensor
    atb::Tensor pvDescale = CreateTensorFromVector(contextPtr, stream, std::vector<float>(headNum, 0), ACL_FLOAT,
                                                   aclFormat::ACL_FORMAT_ND, {headNum});
    atb::SVector<atb::Tensor> inTensors = {qNope, qRope, ctKV, kRope, blockTables, contextLens, qkDescale, pvDescale};
    return inTensors;
}

/**
 * @brief 创建一个MultiLatentAttentionOperation operation
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *CreateMultiLatentAttentionOperation(int headNum)
{
    atb::infer::MultiLatentAttentionParam param;
    param.headNum = headNum;
    param.qkScale = 0.0416666679084301;
    param.kvHeadNum = 1;
    param.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE;
    atb::Operation *mlaOp = nullptr;
    atb::CreateOperation(param, &mlaOp);
    return mlaOp;
}

/**
 * @brief 进行MlaPreprocessOperation的循环调用
 * @param context context指针
 * @param stream stream
 * @param dtype 指定部分输入/输出vector数据类型
 * @param tokenNum 词元数
 * @param headNum 头数
 * @param kSeqLen key/value 的单个词元长度
 */
void RunDemo(atb::Context *context, void *stream, aclDataType dtype, int tokenNum, int headNum, int kSeqLen)
{
    // 创建op
    atb::Operation *mlaOp = CreateMultiLatentAttentionOperation(headNum);

    // 准备输入tensor
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareInTensor(context, stream, dtype, tokenNum, headNum, kSeqLen); // 放入输入tensor
    // 准备输出tensor
    atb::Tensor attenOut = CreateTensor(dtype, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 512});
    variantPack.outTensors = {attenOut}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(mlaOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    for (size_t i = 0; i < 2; i++) {
        std::cout << "tokenNum: " << tokenNum << " headNum: " << headNum << " loop: " << i << std::endl;
        // mlaPreprocess执行
        mlaOp->Execute(variantPack, workspacePtr, workspaceSize, context);
        CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    }
    // 释放资源
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
        for (atb::Tensor &outTensor : variantPack.outTensors) {
            if (outTensor.deviceData == inTensor.deviceData) {
                outTensor.deviceData = nullptr;
            }
        }
        inTensor.deviceData = nullptr;
    }
    for (atb::Tensor &outTensor : variantPack.outTensors) {
        if (outTensor.deviceData == nullptr)
            continue;
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(mlaOp)); // operation，对象概念，先释放
}

int main(int argc, char **argv)
{
    std::string dtypeStr;
    int tokenNum = 4;
    int headNum = 128;
    int kSeqLen = 1500;
    aclDataType dtype = ACL_FLOAT16;
    if (argc == 5) {
        dtypeStr = argv[1];
        tokenNum = std::stoi(argv[2]);
        headNum = std::stoi(argv[3]);
        kSeqLen = std::stoi(argv[4]);
    }
    if (dtypeStr == "bf16") {
        dtype = ACL_BF16;
    }
    // 设置卡号、创建context、设置stream
    atb::Context *context = nullptr;
    void *stream = nullptr;

    CHECK_STATUS(aclInit(nullptr));
    CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
    CHECK_STATUS(atb::CreateContext(&context));
    CHECK_STATUS(aclrtCreateStream(&stream));
    CHECK_STATUS(context->SetExecuteStream(stream));
    // 执行demo
    RunDemo(context, stream, dtype, tokenNum, headNum, kSeqLen);
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "MultiLatentAttention demo success!" << std::endl;
    return 0;
}
