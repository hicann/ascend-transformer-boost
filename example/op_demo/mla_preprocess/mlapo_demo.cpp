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

const int32_t DEVICE_ID = 2;
const uint32_t blockSize = 128;
const uint32_t blockNum = 64;

/**
 * @brief 准备atb::VariantPack中的输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::SVector<atb::Tensor> *atb::VariantPack中的输入tensor
 */
void PrepareInTensor1(atb::Context *contextPtr, aclrtStream stream, aclDataType dtype, int tokenNum,
                      atb::SVector<atb::Tensor> *inTensors)
{
    // 创建shape为[tokenNum, 7168]的输入input tensor
    atb::Tensor input = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * 7168, 0), dtype,
                                               aclFormat::ACL_FORMAT_ND, {tokenNum, 7168});
    // 创建shape为[7168]的输入gamma0 tensor
    atb::Tensor gamma0 = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(7168, 0), dtype,
                                                aclFormat::ACL_FORMAT_ND, {7168});
    // 创建shape为[7168]的输入beta0 tensor
    atb::Tensor beta0 = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(7168, 0), dtype,
                                               aclFormat::ACL_FORMAT_ND, {7168});
    // 创建shape为[1]的输入quantScale0 tensor
    atb::Tensor quantScale0 =
        CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND, {1});
    // 创建shape为[1]的输入quantOffset0 tensor
    atb::Tensor quantOffset0 =
        CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(1, 1), ACL_INT8, aclFormat::ACL_FORMAT_ND, {1});
    // 创建shape为[1,224,2112,32]的输入wdqkv tensor
    atb::Tensor wdqkv = CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(224 * 2112 * 32, 1), ACL_INT8,
                                               aclFormat::ACL_FORMAT_FRACTAL_NZ, {1, 224, 2112, 32});
    // 创建shape为[2112]的输入deScale0 tensor
    atb::Tensor deScale0;
    if (dtype == ACL_BF16) {
        deScale0 = CreateTensorFromVector(contextPtr, stream, std::vector<float>(2112, 1), ACL_FLOAT,
                                          aclFormat::ACL_FORMAT_ND, {2112});
    } else {
        deScale0 = CreateTensorFromVector(contextPtr, stream, std::vector<int64_t>(2112, 10), ACL_INT64,
                                          aclFormat::ACL_FORMAT_ND, {2112});
    }
    // 创建shape为[2112]的输入bias0 tensor
    atb::Tensor bias0 = CreateTensorFromVector(contextPtr, stream, std::vector<int32_t>(2112, 1), ACL_INT32,
                                               aclFormat::ACL_FORMAT_ND, {2112});
    // 创建shape为[1536]的输入gamma1 tensor
    atb::Tensor gamma1 = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1536, 0), dtype,
                                                aclFormat::ACL_FORMAT_ND, {1536});
    // 创建shape为[1536]的输入beta1 tensor
    atb::Tensor beta1 = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1536, 0), dtype,
                                               aclFormat::ACL_FORMAT_ND, {1536});
    // 创建shape为[1]的输入quantScale1 tensor
    atb::Tensor quantScale1 =
        CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND, {1});
    // 创建shape为[1]的输入quantOffset1 tensor
    atb::Tensor quantOffset1 =
        CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(1, 1), ACL_INT8, aclFormat::ACL_FORMAT_ND, {1});
    *inTensors = {input,    gamma0, beta0,  quantScale0, quantOffset0, wdqkv,
                  deScale0, bias0,  gamma1, beta1,       quantScale1,  quantOffset1};
}

/**
 * @brief 准备atb::VariantPack中的输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::SVector<atb::Tensor> *atb::VariantPack中的输入tensor
 */
void PrepareInTensor2(atb::Context *contextPtr, aclrtStream stream, aclDataType dtype, int tokenNum, int headNum,
                      atb::SVector<atb::Tensor> *inTensors)
{
    // 创建shape为[1,48,headNum*192,32]的输入wuq tensor
    atb::Tensor wuq = CreateTensorFromVector(contextPtr, stream, std::vector<int8_t>(48 * headNum * 192 * 32, 1),
                                             ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, {1, 48, headNum * 192, 32});
    // 创建shape为[headNum*192]的输入deScale1 tensor
    atb::Tensor deScale1;
    if (dtype == ACL_BF16) {
        deScale1 = CreateTensorFromVector(contextPtr, stream, std::vector<float>(headNum * 192, 1), ACL_FLOAT,
                                          aclFormat::ACL_FORMAT_ND, {headNum * 192});
    } else {
        deScale1 = CreateTensorFromVector(contextPtr, stream, std::vector<int64_t>(headNum * 192, 10), ACL_INT64,
                                          aclFormat::ACL_FORMAT_ND, {headNum * 192});
    }
    // 创建shape为[headNum*192]的输入bias1 tensor
    atb::Tensor bias1 = CreateTensorFromVector(contextPtr, stream, std::vector<int32_t>(headNum * 192, 1), ACL_INT32,
                                               aclFormat::ACL_FORMAT_ND, {headNum * 192});
    // 创建shape为[512]的输入gamma2 tensor
    atb::Tensor gamma2 =
        CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(512, 0), dtype, aclFormat::ACL_FORMAT_ND, {512});
    // 创建shape为[tokenNum,64]的输入cos tensor
    atb::Tensor cos = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * 64, 0), dtype,
                                             aclFormat::ACL_FORMAT_ND, {tokenNum, 64});
    // 创建shape为[tokenNum,64]的输入sin tensor
    atb::Tensor sin = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * 64, 0.5), dtype,
                                             aclFormat::ACL_FORMAT_ND, {tokenNum, 64});
    // 创建shape为[headNum,32,128,16]的输入wuk tensor
    atb::Tensor wuk = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(headNum * 32 * 128 * 16, 0), dtype,
                                             aclFormat::ACL_FORMAT_FRACTAL_NZ, {headNum, 32, 128, 16});
    // 创建shape为[blockNum, headNum*512/32,block_size, 32]的输入kvCache tensor
    atb::Tensor kvCache = CreateTensorFromVector(
        contextPtr, stream, std::vector<int8_t>(blockNum * headNum * 512 * blockSize, 1), ACL_INT8,
        aclFormat::ACL_FORMAT_FRACTAL_NZ, {blockNum, headNum * 512 / 32, blockSize, 32});
    // 创建shape为[blockNum, headNum*64/16 ,block_size, 16]的输入kvCacheRope tensor
    atb::Tensor kvCacheRope = CreateTensorFromVector(
        contextPtr, stream, std::vector<__fp16>(blockNum * headNum * 64 / 16 * blockSize * 16, 0), dtype,
        aclFormat::ACL_FORMAT_FRACTAL_NZ, {blockNum, headNum * 64 / 16, blockSize, 16});
    auto slotmappingHost = std::vector<int32_t>(1, tokenNum);
    for (size_t i = 0; i < slotmappingHost.size(); i++)
        slotmappingHost[i] = static_cast<int32_t>(i);
    // 创建shape为[headNum]的输入slotmapping tensor
    atb::Tensor slotmapping =
        CreateTensorFromVector(contextPtr, stream, slotmappingHost, ACL_INT32, aclFormat::ACL_FORMAT_ND, {tokenNum});
    // 创建shape为[1]的输入ctkvScale tensor
    atb::Tensor ctkvScale =
        CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(1, 0), dtype, aclFormat::ACL_FORMAT_ND, {1});
    // 创建shape为[headNum]的输入qNopeScale tensor
    atb::Tensor qNopeScale = CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(headNum, 0), dtype,
                                                    aclFormat::ACL_FORMAT_ND, {headNum});
    atb::SVector<atb::Tensor> tempTensors = {wuq, deScale1, bias1,       gamma2,      cos,       sin,
                                             wuk, kvCache,  kvCacheRope, slotmapping, ctkvScale, qNopeScale};

    for (auto &tensor : tempTensors) {
        inTensors->push_back(tensor);
    }
}

/**
 * @brief 创建一个MlaPreprocessOperation operation
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *CreateMlaPreprocessOperation()
{
    atb::infer::MlaPreprocessParam param;
    param.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE;
    atb::Operation *mlaPreprocessOp = nullptr;
    atb::CreateOperation(param, &mlaPreprocessOp);
    return mlaPreprocessOp;
}

/**
 * @brief 进行MlaPreprocessOperation的循环调用
 * @param context context指针
 * @param stream stream
 * @param dtype 指定部分输入/输出vector数据类型
 * @param tokenNum 词元数
 * @param headNum 头数
 */
void RunDemo(atb::Context *context, void *stream, aclDataType dtype, int tokenNum, int headNum)
{
    // 创建op
    atb::Operation *mlaPreprocessOp = CreateMlaPreprocessOperation();
    // 准备输入tensor
    atb::VariantPack variantPack;
    // 放入rmsNormQuant_0，matmul_0，rmsNormQuant_1输入tensor
    PrepareInTensor1(context, stream, dtype, tokenNum, &variantPack.inTensors);
    // 放入matmul_1，rmsNorm，rope，matmulEin，reshapeAndCache，quant输入tensor
    PrepareInTensor2(context, stream, dtype, tokenNum, headNum, &variantPack.inTensors);
    // 准备输出tensor
    atb::Tensor qOut0 = CreateTensor(ACL_INT8, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 512});
    atb::Tensor &kvCacheOut0 = variantPack.inTensors.at(19);
    atb::Tensor qOut1 = CreateTensor(dtype, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 64});
    atb::Tensor &kvCacheOut1 = variantPack.inTensors.at(20);
    variantPack.outTensors = {qOut0, kvCacheOut0, qOut1, kvCacheOut1}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(mlaPreprocessOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    for (size_t i = 0; i < 10; i++) {
        std::cout << "tokenNum: " << tokenNum << " headNum: " << headNum << " loop: " << i << std::endl;
        // mlaPreprocess执行
        mlaPreprocessOp->Execute(variantPack, workspacePtr, workspaceSize, context);
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
    CHECK_STATUS(atb::DestroyOperation(mlaPreprocessOp)); // operation，对象概念，先释放
}


int main(int argc, char **argv)
{
    std::string dtypeStr;
    int tokenNum = 4;
    int headNum = 128;
    aclDataType dtype = ACL_FLOAT16;
    if (argc == 4) {
        dtypeStr = argv[1];
        tokenNum = std::stoi(argv[2]);
        headNum = std::stoi(argv[3]);
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
