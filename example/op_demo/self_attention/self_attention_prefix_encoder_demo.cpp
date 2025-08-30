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
#include <cmath>

const uint32_t BATCH_SIZE = 4;                      // 批处理大小
std::vector<int32_t> seqLenHost = {16, 16, 32, 32}; // host侧tensor值，用于存储Query每个批处理中的序列长度
const uint32_t NTOKENS = accumulate(seqLenHost.begin(), seqLenHost.end(), 0); // sum(seqLenHost)
std::vector<int32_t> kvSeqLenHost = {16, 144, 32, 288}; // host侧tensor值，用于存储Key, Value每个批处理中的序列长度
const uint32_t NUM_BLOCKS = accumulate(kvSeqLenHost.begin(), kvSeqLenHost.end(), 0); // sum(kvSeqLenHost)
const uint32_t HEAD_NUM = 32;                                                        // query头数
const uint32_t KV_HEAD_NUM = 8;                                                      // kv头数
const uint32_t HEAD_SIZE = 128;                                                      // 头大小
const uint32_t BLOCK_SIZE = 128;                                                     // 以block存放的kv块大小

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param seqLenHost host侧tensor。Query序列长度向量
 * @param kvSeqLenHost host侧tensor。Key, Value序列长度向量
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status atb错误码
 * @note 需要传入所有host侧tensor
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, std::vector<int32_t> &seqLenHost,
                            std::vector<int32_t> &kvSeqLenHost, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建query tensor
    atb::Tensor tensorQ;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(NTOKENS * HEAD_NUM * HEAD_SIZE, 1.0),
                                        ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE},
                                        tensorQ));
    // 创建key，value tensor
    std::vector<float> kvData(NUM_BLOCKS * BLOCK_SIZE * KV_HEAD_NUM * HEAD_SIZE, 1.0);
    std::vector<int64_t> kvShape = {NUM_BLOCKS, BLOCK_SIZE, KV_HEAD_NUM, HEAD_SIZE};
    atb::Tensor tensorK;
    CHECK_STATUS(
        CreateTensorFromVector(contextPtr, stream, kvData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, kvShape, tensorK));
    atb::Tensor tensorV;
    CHECK_STATUS(
        CreateTensorFromVector(contextPtr, stream, kvData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, kvShape, tensorV));
    // 创建blockTables
    atb::Tensor tensorBlockTables;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE, 4}, tensorBlockTables));
    std::vector<int32_t> blockTablesData(16);
    std::iota(blockTablesData.begin(), blockTablesData.end(), 0);
    CHECK_STATUS(aclrtMemcpy(tensorBlockTables.deviceData, tensorBlockTables.dataSize, blockTablesData.data(),
                             sizeof(int32_t) * blockTablesData.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    // 创建alibi128mask，开启高精度后mask填充值使用1替代-inf
    std::vector<float> maskData = std::vector<float>(HEAD_NUM * NTOKENS * 128, 0); // alibi128 mask
    for (int i = 0; i < HEAD_NUM; ++i) {
        for (int j = 0; j < NTOKENS; ++j) {
            for (int k = j + 1; k < 128; ++k) {
                maskData[i * NTOKENS * 128 + j * 128 + k] = 1;
            }
        }
    }
    atb::Tensor tensorMask;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, maskData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {HEAD_NUM, NTOKENS, 128}, tensorMask));
    // 创建seqLen，host侧tensor
    atb::Tensor tensorSeqLen;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE}, tensorSeqLen));
    tensorSeqLen.hostData = seqLenHost.data();
    // 创建kvSeqLen，host侧tensor
    atb::Tensor tensorKvSeqLen;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE}, tensorKvSeqLen));
    tensorKvSeqLen.hostData = kvSeqLenHost.data();
    atb::Tensor tensorSlopes;
    CHECK_STATUS(CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {HEAD_SIZE}, tensorSlopes));
    std::vector<float> slData(HEAD_SIZE, 1.0);
    CHECK_STATUS(aclrtMemcpy(tensorSlopes.deviceData, tensorSlopes.dataSize, slData.data(),
                             sizeof(float) * slData.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    inTensors = {tensorQ, tensorK, tensorV, tensorBlockTables, tensorMask, tensorSeqLen, tensorKvSeqLen, tensorSlopes};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个FA encoder的Operation，并设置参数
 * @param 创建一个Operation指针
 * @return atb::Status atb错误码
 */
atb::Status PrepareOperation(atb::Operation **prefixEncoderOp)
{
    atb::infer::SelfAttentionParam prefixOpParam;
    prefixOpParam.quantType = atb::infer::SelfAttentionParam::QuantType::TYPE_QUANT_UNQUANT; // 非量化场景
    prefixOpParam.outDataType = ACL_DT_UNDEFINED; // 非量化场景，不设置输出类型
    prefixOpParam.headNum = HEAD_NUM;             // query 头数
    prefixOpParam.kvHeadNum = KV_HEAD_NUM;        // key, value 头数
    prefixOpParam.qScale = 1;                     // query缩放系数，不缩放置1
    prefixOpParam.qkScale = 1 / sqrt(HEAD_SIZE);  // tor值，Q*K^T后的缩放系数，不缩放置1
    prefixOpParam.batchRunStatusEnable = false;   // 不开启动态batch
    prefixOpParam.isTriuMask = 1; // 是否开启mask倒三角优化，这里开启，和压缩mask一起使用
    // 计算类型/场景分类，使用Prefix Encoder
    prefixOpParam.calcType = atb::infer::SelfAttentionParam::CalcType::PREFIX_ENCODER;
    // 高精度，softmax使用float32
    prefixOpParam.kernelType = atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_HIGH_PRECISION;
    prefixOpParam.clampType = atb::infer::SelfAttentionParam::ClampType::CLAMP_TYPE_UNDEFINED; // 不开启最大最小值截断
    prefixOpParam.clampMin = 0;                                                                // 不开启截断时置0
    prefixOpParam.clampMax = 0;                                                                // 不开启截断时置0
    prefixOpParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS; // 使用128x128的上三角
    prefixOpParam.kvcacheCfg = atb::infer::SelfAttentionParam::KvCacheCfg::K_CACHE_V_CACHE; // 会进行kvCache处理
    prefixOpParam.scaleType = atb::infer::SelfAttentionParam::ScaleType::SCALE_TYPE_TOR; // 缩放类型，使用qkScale缩放
    prefixOpParam.inputLayout = atb::infer::InputLayout::TYPE_BSND;                      // 数据排布格式，BNSD
    prefixOpParam.mlaVHeadSize = 0; // 不开启MLA合并kvCache，置0
    prefixOpParam.cacheType = atb::infer::SelfAttentionParam::CacheType::CACHE_TYPE_NORM; // 不开启SWA mask
    prefixOpParam.windowSize = 0;                                                         // 不开启SWA mask，置0
    return atb::CreateOperation(prefixOpParam, prefixEncoderOp);
}

int main(int argc, char **argv)
{
    if (!Is910B()) {
        std::cout << "Self attention PrefixEncoder demo only supports Atlas A2/A3 products" << std::endl;
        return 0;
    }
    // 设置卡号、创建context、设置stream
    CHECK_STATUS(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    CHECK_STATUS(context->SetExecuteStream(stream));

    // FA Prefix Encoder示例
    atb::Operation *prefixEncoderOp = nullptr;
    CHECK_STATUS(PrepareOperation(&prefixEncoderOp));
    // 准备输入tensor
    atb::VariantPack prefixVariantPack;
    CHECK_STATUS(
        PrepareInTensor(context, stream, seqLenHost, kvSeqLenHost, prefixVariantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE}, tensorOut));
    prefixVariantPack.outTensors = {tensorOut}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(prefixEncoderOp->Setup(prefixVariantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // FA Prefix Encoder执行
    CHECK_STATUS(prefixEncoderOp->Execute(prefixVariantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    CHECK_STATUS(aclrtFree(tensorOut.deviceData));
    for (atb::Tensor &inTensor : prefixVariantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(prefixEncoderOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "FA Prefix Encoder demo success!" << std::endl;
    return 0;
}
