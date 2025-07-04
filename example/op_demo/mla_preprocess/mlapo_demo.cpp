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

namespace {
const int32_t DEVICE_ID = 2;
const uint32_t BLOCK_SIZE = 128;
const uint32_t BLOCK_NUM = 64;

const int32_t INPUT_NUM = 4;
const int32_t DTYPE_IDX = 1;
const int32_t TOKEN_NUM_IDX = 2;
const int32_t HEAD_NUM_IDX = 3;
const int32_t KV_CACHE_NOPE_IDX = 19;
const int32_t KV_CACHE_ROPE_IDX = 20;


const int32_t RUNS = 10;

const int32_t ROPE_DIM64 = 64;
const int32_t NZ_DIM16 = 16;
const int32_t NOPE_DIM512 = 512;
const int32_t RMSNORM_QUANT_DIM7168 = 7168;
const int32_t RMSNORM_QUANT_DIM1536 = 1536;
const int32_t MATMUL_DIM224 = 224;
const int32_t MATMUL_DIM2112 = 2112;
const int32_t MATMUL_DIM32 = 32;
const int32_t MATMUL_DIM192 = 192;
const int32_t MATMUL_DIM48 = 48;
}

/**
 * @brief 准备atb::VariantPack中的输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::SVector<atb::Tensor> *atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor1(atb::Context *contextPtr, aclrtStream stream, aclDataType dtype, int tokenNum,
                             atb::SVector<atb::Tensor> *inTensors)
{
    // 创建shape为[tokenNum, 7168]的输入input tensor
    atb::Tensor input;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * RMSNORM_QUANT_DIM7168, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {tokenNum, RMSNORM_QUANT_DIM7168}, input, dtype));
    // 创建shape为[7168]的输入gamma0 tensor
    atb::Tensor gamma0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(RMSNORM_QUANT_DIM7168, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {RMSNORM_QUANT_DIM7168}, gamma0, dtype));
    // 创建shape为[7168]的输入beta0 tensor
    atb::Tensor beta0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(RMSNORM_QUANT_DIM7168, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {RMSNORM_QUANT_DIM7168}, beta0, dtype));
    // 创建shape为[1]的输入quantScale0 tensor
    atb::Tensor quantScale0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND,
                                        {1}, quantScale0, dtype));
    // 创建shape为[1]的输入quantOffset0 tensor
    atb::Tensor quantOffset0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(1, 1), ACL_INT8,
                                        aclFormat::ACL_FORMAT_ND, {1}, quantOffset0));
    // 创建shape为[1,224,2112,32]的输入wdqkv tensor
    atb::Tensor wdqkv;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(MATMUL_DIM224 * MATMUL_DIM2112 * MATMUL_DIM32, 1), ACL_INT8,
                                        aclFormat::ACL_FORMAT_FRACTAL_NZ, {1, MATMUL_DIM224, MATMUL_DIM2112, MATMUL_DIM32}, wdqkv));
    // 创建shape为[2112]的输入deScale0 tensor
    atb::Tensor deScale0;
    if (dtype == ACL_BF16) {
        CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(MATMUL_DIM2112, 1), ACL_FLOAT,
                                            aclFormat::ACL_FORMAT_ND, {MATMUL_DIM2112}, deScale0));
    } else {
        CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int64_t>(MATMUL_DIM2112, 10), ACL_INT64,
                                            aclFormat::ACL_FORMAT_ND, {MATMUL_DIM2112}, deScale0));
    }
    // 创建shape为[2112]的输入bias0 tensor
    atb::Tensor bias0;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int32_t>(MATMUL_DIM2112, 1), ACL_INT32,
                                        aclFormat::ACL_FORMAT_ND, {MATMUL_DIM2112}, bias0));
    // 创建shape为[1536]的输入gamma1 tensor
    atb::Tensor gamma1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(RMSNORM_QUANT_DIM1536, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {RMSNORM_QUANT_DIM1536}, gamma1, dtype));
    // 创建shape为[1536]的输入beta1 tensor
    atb::Tensor beta1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(RMSNORM_QUANT_DIM1536, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {RMSNORM_QUANT_DIM1536}, beta1, dtype));
    // 创建shape为[1]的输入quantScale1 tensor
    atb::Tensor quantScale1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND,
                                        {1}, quantScale1, dtype));
    // 创建shape为[1]的输入quantOffset1 tensor
    atb::Tensor quantOffset1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(1, 1), ACL_INT8,
                                        aclFormat::ACL_FORMAT_ND, {1}, quantOffset1));
    *inTensors = {input,    gamma0, beta0,  quantScale0, quantOffset0, wdqkv,
                  deScale0, bias0,  gamma1, beta1,       quantScale1,  quantOffset1};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 准备atb::VariantPack中的输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::SVector<atb::Tensor> *atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor2(atb::Context *contextPtr, aclrtStream stream, aclDataType dtype, int tokenNum, int headNum,
                             atb::SVector<atb::Tensor> *inTensors)
{
    // 创建shape为[1,48,headNum*192,32]的输入wuq tensor
    atb::Tensor wuq;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(MATMUL_DIM48 * headNum * MATMUL_DIM192 * MATMUL_DIM32, 1), ACL_INT8,
                                        aclFormat::ACL_FORMAT_FRACTAL_NZ, {1, MATMUL_DIM48, headNum * MATMUL_DIM192, MATMUL_DIM32}, wuq));
    // 创建shape为[headNum*192]的输入deScale1 tensor
    atb::Tensor deScale1;
    if (dtype == ACL_BF16) {
        CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(headNum * MATMUL_DIM192, 1), ACL_FLOAT,
                                            aclFormat::ACL_FORMAT_ND, {headNum * MATMUL_DIM192}, deScale1));
    } else {
        CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int64_t>(headNum * MATMUL_DIM192, 1), ACL_INT64,
                                            aclFormat::ACL_FORMAT_ND, {headNum * MATMUL_DIM192}, deScale1));
    }
    // 创建shape为[headNum*192]的输入bias1 tensor
    atb::Tensor bias1;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<int32_t>(headNum * MATMUL_DIM192, 1), ACL_INT32,
                                        aclFormat::ACL_FORMAT_ND, {headNum * MATMUL_DIM192}, bias1));
    // 创建shape为[512]的输入gamma2 tensor
    atb::Tensor gamma2;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(NOPE_DIM512, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {NOPE_DIM512}, gamma2, dtype));
    // 创建shape为[tokenNum,64]的输入cos tensor
    atb::Tensor cos;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * ROPE_DIM64, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {tokenNum, ROPE_DIM64}, cos, dtype));
    // 创建shape为[tokenNum,64]的输入sin tensor
    atb::Tensor sin;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * ROPE_DIM64, 0.5), dtype,
                                        aclFormat::ACL_FORMAT_ND, {tokenNum, ROPE_DIM64}, sin, dtype));
    // 创建shape为[headNum,32,128,16]的输入wuk tensor
    atb::Tensor wuk;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(headNum * MATMUL_DIM32 * BLOCK_SIZE * NZ_DIM16, 0), dtype,
                                        aclFormat::ACL_FORMAT_FRACTAL_NZ, {headNum, MATMUL_DIM32, BLOCK_SIZE, NZ_DIM16}, wuk, dtype));
    // 创建shape为[BLOCK_NUM, headNum*512/32,block_size, 32]的输入kvCache tensor
    atb::Tensor kvCache;
    CHECK_STATUS(CreateTensorFromVector(
        contextPtr, stream, std::vector<int8_t>(BLOCK_NUM * headNum * NOPE_DIM512 * BLOCK_SIZE, 1), ACL_INT8,
        aclFormat::ACL_FORMAT_FRACTAL_NZ, {BLOCK_NUM, headNum * NOPE_DIM512 / MATMUL_DIM32, BLOCK_SIZE, MATMUL_DIM32}, kvCache));
    // 创建shape为[BLOCK_NUM, headNum*64/16 ,block_size, 16]的输入kvCacheRope tensor
    atb::Tensor kvCacheRope;
    CHECK_STATUS(CreateTensorFromVector(
        contextPtr, stream, std::vector<__fp16>(BLOCK_NUM * headNum * ROPE_DIM64 / NZ_DIM16 * BLOCK_SIZE * NZ_DIM16, 0), dtype,
        aclFormat::ACL_FORMAT_FRACTAL_NZ, {BLOCK_NUM, headNum * ROPE_DIM64 / NZ_DIM16, BLOCK_SIZE, NZ_DIM16}, kvCacheRope, dtype));
    auto slotmappingHost = std::vector<int32_t>(1, tokenNum);
    for (size_t i = 0; i < slotmappingHost.size(); i++)
        slotmappingHost[i] = static_cast<int32_t>(i);
    // 创建shape为[headNum]的输入slotmapping tensor
    atb::Tensor slotmapping;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, slotmappingHost, ACL_INT32, aclFormat::ACL_FORMAT_ND,
                                        {tokenNum}, slotmapping));
    // 创建shape为[1]的输入ctkvScale tensor
    atb::Tensor ctkvScale;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND,
                                        {1}, ctkvScale, dtype));
    // 创建shape为[headNum]的输入qNopeScale tensor
    atb::Tensor qNopeScale;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(headNum, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {headNum}, qNopeScale, dtype));
    atb::SVector<atb::Tensor> tempTensors = {wuq, deScale1, bias1,       gamma2,      cos,       sin,
                                             wuk, kvCache,  kvCacheRope, slotmapping, ctkvScale, qNopeScale};
    for (auto &tensor : tempTensors) {
        inTensors->push_back(tensor);
    }
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个MlaPreprocessOperation operation
 * @param atb::Operation * 返回一个Operation指针
 */
atb::Status CreateMlaPreprocessOperation(atb::Operation **mlaPreprocessOp)
{
    atb::infer::MlaPreprocessParam param;
    param.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE;
    return atb::CreateOperation(param, mlaPreprocessOp);
}

/**
 * @brief 进行MlaPreprocessOperation的循环调用
 * @param context context指针
 * @param stream stream
 * @param dtype 指定部分输入/输出vector数据类型
 * @param tokenNum 词元数
 * @param headNum 头数
 * @return atb::Status 错误码
 */
atb::Status RunDemo(atb::Context *context, void *stream, aclDataType dtype, int tokenNum, int headNum)
{
    // 创建op
    atb::Operation *mlaPreprocessOp = nullptr;
    CHECK_STATUS(CreateMlaPreprocessOperation(&mlaPreprocessOp));
    // 准备输入tensor
    atb::VariantPack variantPack;
    // 放入rmsNormQuant_0，matmul_0，rmsNormQuant_1输入tensor
    CHECK_STATUS(PrepareInTensor1(context, stream, dtype, tokenNum, &variantPack.inTensors));
    // 放入matmul_1，rmsNorm，rope，matmulEin，reshapeAndCache，quant输入tensor
    CHECK_STATUS(PrepareInTensor2(context, stream, dtype, tokenNum, headNum, &variantPack.inTensors));
    // 准备输出tensor
    atb::Tensor qOut0;
    CreateTensor(ACL_INT8, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, NOPE_DIM512}, qOut0);
    atb::Tensor &kvCacheOut0 = variantPack.inTensors.at(KV_CACHE_NOPE_IDX);
    atb::Tensor qOut1;
    CreateTensor(dtype, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, ROPE_DIM64}, qOut1);
    atb::Tensor &kvCacheOut1 = variantPack.inTensors.at(KV_CACHE_ROPE_IDX);
    variantPack.outTensors = {qOut0, kvCacheOut0, qOut1, kvCacheOut1}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(mlaPreprocessOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    for (size_t i = 0; i < RUNS; i++) {
        std::cout << "tokenNum: " << tokenNum << " headNum: " << headNum << " loop: " << i << std::endl;
        // mlaPreprocess执行
        CHECK_STATUS(mlaPreprocessOp->Execute(variantPack, workspacePtr, workspaceSize, context));
        CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    }
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
    return atb::DestroyOperation(mlaPreprocessOp); // operation，对象概念，先释放
}


int main(int argc, char **argv)
{
    std::string dtypeStr;
    int tokenNum = 4;
    int headNum = 128;
    aclDataType dtype = ACL_FLOAT16;
    if (argc == INPUT_NUM) {
        dtypeStr = argv[DTYPE_IDX];
        tokenNum = std::stoi(argv[TOKEN_NUM_IDX]);
        headNum = std::stoi(argv[HEAD_NUM_IDX]);
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
    context->SetExecuteStream(stream);
    RunDemo(context, stream, dtype, tokenNum, headNum);
    // 释放资源
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "MlaPreprocess demo success!" << std::endl;
    return 0;
}
