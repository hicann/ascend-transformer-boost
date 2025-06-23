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

const uint32_t BATCH_SIZE = 100;                 // 批处理大小
std::vector<int32_t> seqLenHost(BATCH_SIZE, 16); // host侧tensor值，用于存储每个批处理中的序列长度
std::vector<int32_t> tokenOffsetHost(BATCH_SIZE, 16);                         // host侧tensor值，token偏移
std::vector<int32_t> layerId(1, 0);                                           // device侧，kvCache中取哪个计算
const uint32_t NTOKENS = accumulate(seqLenHost.begin(), seqLenHost.end(), 0); // sum(seqLenHost)
const uint32_t MAX_SEQ_LEN = 1024;                                            // 最大序列长度
const uint32_t HEAD_NUM = 32;                                                 // 头数
const uint32_t KV_HEAD_NUM = 32;                                              // kv头数
const uint32_t HEAD_SIZE = 64;                                                // 头大小

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
atb::SVector<atb::Tensor> PrepareInTensor(atb::Context *contextPtr, aclrtStream stream,
                                          std::vector<int32_t> &seqLenHost, std::vector<int32_t> &tokenOffsetHost,
                                          std::vector<int32_t> &layerId)
{
    uint32_t qHiddenSize = HEAD_NUM * HEAD_SIZE;
    uint32_t kvHiddenSize = KV_HEAD_NUM * HEAD_SIZE;

    // 创建query tensor
    std::vector<float> qData(NTOKENS * qHiddenSize, 1.0);
    atb::Tensor tensorQ = CreateTensorFromVector(contextPtr, stream, qData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                 {NTOKENS, qHiddenSize});
    // 创建key，value tensor
    std::vector<float> kvData(NTOKENS * kvHiddenSize, 1.0);
    atb::Tensor tensorK = CreateTensorFromVector(contextPtr, stream, kvData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                 {NTOKENS, qHiddenSize});
    atb::Tensor tensorV = CreateTensorFromVector(contextPtr, stream, kvData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                 {NTOKENS, qHiddenSize});
    std::vector<float> kvCacheData(BATCH_SIZE * MAX_SEQ_LEN * kvHiddenSize, 1.0);
    std::vector<int64_t> kvCacheShape = {1, BATCH_SIZE, MAX_SEQ_LEN, kvHiddenSize};
    atb::Tensor tensorCacheK =
        CreateTensorFromVector(contextPtr, stream, kvCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, kvCacheShape);
    atb::Tensor tensorCacheV =
        CreateTensorFromVector(contextPtr, stream, kvCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, kvCacheShape);
    // 创建norm mask，值为-inf的上三角mask
    std::vector<float> maskData(BATCH_SIZE * MAX_SEQ_LEN * MAX_SEQ_LEN, 0);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < MAX_SEQ_LEN; ++j) {
            for (int k = j + 1; k < MAX_SEQ_LEN; ++k) {
                maskData[i * MAX_SEQ_LEN * MAX_SEQ_LEN + j * MAX_SEQ_LEN + k] = -32768;
            }
        }
    }
    atb::Tensor tensorMask = CreateTensorFromVector(contextPtr, stream, maskData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                                    {BATCH_SIZE, MAX_SEQ_LEN, MAX_SEQ_LEN});
    // 创建tokenOffset，host侧tensor
    atb::Tensor tensorTokenOffset = CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE});
    tensorTokenOffset.hostData = tokenOffsetHost.data(); // host侧tensor，拷贝值
    // 创建seqLen，host侧tensor
    atb::Tensor tensorSeqLen = CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE});
    tensorSeqLen.hostData = seqLenHost.data(); // host侧tensor，拷贝值
    // 创建layerId
    atb::Tensor tensorLayerId = CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {1});
    CHECK_STATUS(aclrtMemcpy(tensorLayerId.deviceData, tensorLayerId.dataSize, layerId.data(),
                             sizeof(short) * layerId.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    // 根据顺序将所有输入tensor放入SVector
    atb::SVector<atb::Tensor> inTensors = {tensorQ,    tensorK,           tensorV,      tensorCacheK, tensorCacheV,
                                           tensorMask, tensorTokenOffset, tensorSeqLen, tensorLayerId};
    return inTensors;
}

/**
 * @brief 创建一个FA encoder的Operation，并设置参数
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *PrepareOperation()
{
    atb::infer::SelfAttentionParam opParam;
    opParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    opParam.headNum = HEAD_NUM;
    opParam.kvHeadNum = KV_HEAD_NUM;
    opParam.calcType = atb::infer::SelfAttentionParam::CalcType::ENCODER;
    atb::Operation *encoderOp = nullptr;
    CHECK_STATUS(atb::CreateOperation(opParam, &encoderOp));
    return encoderOp;
}

int main(int argc, char **argv)
{
    // kv隐藏层大小，用于输出tensor shape
    uint32_t kvHiddenSize = KV_HEAD_NUM * HEAD_SIZE;
    // 设置卡号、创建context、设置stream
    CHECK_STATUS(aclInit(nullptr));
    if (!Is910B()) {
        std::cout << "This self attention demo only supports Atlas A2/A3 products" << std::endl;
        return 0;
    }
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // FA Encoder示例
    atb::Operation *encoderOp = PrepareOperation();
    // 准备输入张量
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareInTensor(context, stream, seqLenHost, tokenOffsetHost, layerId); // 放入输入tensor
    atb::Tensor tensorOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, kvHiddenSize});
    variantPack.outTensors.push_back(tensorOut); // 放入输出tensor
    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(encoderOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // FA Encoder执行
    encoderOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    CHECK_STATUS(aclrtFree(tensorOut.deviceData));
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(encoderOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "FA Encoder demo for Atlas A2/A3 success!" << std::endl;
    return 0;
}
