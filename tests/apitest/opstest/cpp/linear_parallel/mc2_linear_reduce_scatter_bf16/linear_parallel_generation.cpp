/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <acl/acl.h>
#include "../../demo_util.h"

const int32_t DEV_NUM = 2;

const int32_t M = 2;
const int32_t K = 256;
const int32_t N = 2;

atb::Status saveTensor(atb::Tensor tensor, std::string path)
{
    if (tensor.deviceData == nullptr) {}
    void *hostData = nullptr;
    aclrtMallocHost((void **)&hostData, tensor.dataSize);
    aclrtMemcpy(hostData, tensor.dataSize, tensor.deviceData, tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<char *>(hostData), tensor.dataSize);
    file.close();
    aclrtFreeHost(hostData);
    return atb::ErrorType::NO_ERROR;
}

atb::Status ExcuteImpl(atb::Operation *op, atb::VariantPack variantPack, atb::Context *context, aclrtStream &stream)
{
    uint64_t workspaceSize = 0;
    CHECK_STATUS(op->Setup(variantPack, workspaceSize, context));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_STATUS(op->Execute(variantPack, (uint8_t *)workspace, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    if (workspace) {
        CHECK_STATUS(aclrtFree(workspace)); // 销毁workspace
    }
    return atb::ErrorType::NO_ERROR;
}

atb::Status LinearParallelOneThread(int rank, int rankSize, atb::Context *&context, aclrtStream &stream)
{
    int deviceId = rank;
    CHECK_STATUS(aclrtSetDevice(deviceId));

    atb::Tensor input;
    CHECK_STATUS(CreateTensorFromVector(context, stream, std::vector<float>(512 * 512, 2.0), aclDataType::ACL_BF16,
                                        aclFormat::ACL_FORMAT_ND, {512, 512}, input));
    atb::Tensor weight;
    CHECK_STATUS(CreateTensorFromVector(context, stream, std::vector<float>(512 * 512, 2.0), aclDataType::ACL_BF16,
                                        aclFormat::ACL_FORMAT_ND, {512, 512}, weight));

    atb::Tensor output;
    output.desc.dtype = ACL_BF16;
    output.desc.format = ACL_FORMAT_ND;
    output.desc.shape.dimNum = 2;
    output.desc.shape.dims[0] = 256;
    output.desc.shape.dims[1] = 512;
    output.dataSize = atb::Utils::GetTensorSize(output);
    CHECK_STATUS(aclrtMalloc(&output.deviceData, output.dataSize, ACL_MEM_MALLOC_HUGE_FIRST));

    atb::infer::LinearParallelParam param;
    param.transWeight = false;
    param.rank = rank;
    param.rankRoot = 0;
    param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
    param.rankSize = rankSize;
    param.backend = "mc2";
    param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER;
    atb::Operation *op = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &op));

    atb::VariantPack variantPack;
    variantPack.inTensors = {input, weight};
    variantPack.outTensors = {output};
    ExcuteImpl(op, variantPack, context, stream);
    std::cout << "rank: " << rank << " executed END." << std::endl;
    saveTensor(input, "rank" + std::to_string(rank) + "_inTensor0.bin");
    saveTensor(weight, "rank" + std::to_string(rank) + "_inTensor1.bin");
    saveTensor(output, "rank" + std::to_string(rank) + "_outTensor0.bin");
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(op));    // 销毁op对象
    CHECK_STATUS(aclrtDestroyStream(stream));   // 销毁stream
    CHECK_STATUS(atb::DestroyContext(context)); // 销毁context
    return atb::ErrorType::NO_ERROR;
}

int main(int argc, const char *argv[])
{
    int ret = aclInit(nullptr);
    atb::Context *context[DEV_NUM] = {nullptr};
    aclrtStream stream[DEV_NUM] = {nullptr};
    for (size_t i = 0; i < DEV_NUM; i++) {
        aclrtSetDevice(i);
        atb::CreateContext(&context[i]);
        aclrtCreateStream(&stream[i]);
        context[i]->SetExecuteStream(stream[i]);
    }

    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (size_t i = 0; i < DEV_NUM; i++) {
        threads[i].reset(new (std::nothrow) std::thread(LinearParallelOneThread, i, DEV_NUM, std::ref(context[i]),
                                                        std::ref(stream[i])));
    }
    for (size_t i = 0; i < DEV_NUM; ++i) {
        threads[i]->join();
    }
    CHECK_STATUS(aclFinalize());
    return 0;
}
