/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "c_interface_utils.h"

#include <algorithm>
#include <iostream>
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

using namespace atb;
using namespace atb::cinterfaceTest;

const int64_t INOUT_TENSOR_NUM = 9;
const int64_t SEQLEN_INDEX = 5;
const int64_t KV_SEQLEN_INDEX = 6;

void TestSelfAttentionPrefixEncoder(const int64_t headNum, const int64_t kvHeadNum, const int64_t headSize,
                                    const int64_t batch, const int64_t numBlocks, const int64_t blockSize,
                                    const float qkscale, const int maskType, const aclDataType dtype,
                                    const int64_t qSeqLen)
{
    if (!GetSingleton<Config>().Is910B()) {
        std::cout << "SelfAttentionPrefixEncoder only supports A2/A3" << std::endl;
        exit(0);
    }
    int inputNum = INOUT_TENSOR_NUM;
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    Init(&context, &stream, &deviceId);
    uint8_t *inoutHost[inputNum];
    uint8_t *inoutDevice[inputNum];
    aclTensor *tensorList[inputNum];
    std::vector<aclDataType> inputTypes = {dtype, dtype, dtype, ACL_INT32, ACL_INT32, ACL_INT32, dtype, ACL_FLOAT, dtype};
    if (dtype == ACL_BF16) {
        inputTypes[4] = ACL_UINT32;
        inputTypes[5] = ACL_UINT32;
    }

    int64_t queryDim[batch + 1];
    for (int i = 0; i < batch; ++i) {
        queryDim[i] = qSeqLen;
    }
    queryDim[batch] = headNum * headSize;
    std::vector<std::vector<int64_t>> tensorDim = {
        queryDim,                                   // query
        {numBlocks, blockSize, headNum * headSize}, // key
        {numBlocks, blockSize, headNum * headSize}, // value
        {batch, maxBlockNum},                       // blockTables
        {batch},                                    // seqLen
        {batch},                                    // kvSeqLen
        {headNum * headSize},                       // slopes
        {256, 256},                                 // mask
        queryDim,                                   // attnOut
    };
    size_t inoutSize[inputNum];
    int total = 0;
    for (int i = 0; i < inputNum; ++i) {
        if (tensorDim[i].size() == 0) {
            inoutSize[i] = 0;
            continue;
        }
        total = 1;
        for (int j = 0; j < tensorDim[i].size(); ++j) {
            total *= tensorDim[i][j];
        }
        inoutSize[i] = total * GetDataTypeSize(inputTypes[i]);
    }
    CreateInOutData(inputNum, inoutHost, inoutDevice, inoutSize);
    size_t i = 0;


    while (i < tensorDim.size()) {
        if (i == SEQLEN_INDEX || i == KV_SEQLEN_INDEX) {
            CreateACLTensorInOut(tensorDim[i], inputTypes[i], ACL_FORMAT_ND, tensorList, i, (void *)seqLen.data());
            continue;
        }
        CreateACLTensorInOut(tensorDim[i], inputTypes[i], ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
    }
    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;

    Status ret = AtbSelfAttentionPrefixEncoderGetWorkspaceSize(
        tensorList[0], tensorList[1], tensorList[2], tensorList[3], tensorList[4], tensorList[5], tensorList[6],
        tensorList[7], maskType, headNum, kvHeadNum, qkScale, tensorList[8], &workspaceSize, &op, context);
    EXPECT_EQ(ret, ACL_ERROR_NONE);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_ERROR_NONE);
    }
    ret = AtbSelfAttentionPrefixEncoder(workspaceAddr, workspaceSize, op, context);
    EXPECT_EQ(ret, ACL_ERROR_NONE);

    ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(ret, ACL_ERROR_NONE);

    if (workspaceSize > 0) {
        EXPECT_EQ(aclrtFree(workspaceAddr), ACL_ERROR_NONE);
    }
    EXPECT_EQ(atb::DestroyOperation(op), NO_ERROR);
    Destroy(&context, &stream);
    for (i = 0; i < MLAINOUTMLA; i++) {
        aclrtFreeHost(inoutHost[i]);
        aclrtFree(inoutDevice[i]);
    }
}
