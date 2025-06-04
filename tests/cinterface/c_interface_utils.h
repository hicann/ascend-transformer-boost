/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef C_INTERFACE_UTILS_H
#define C_INTERFACE_UTILS_H
#include <gtest/gtest.h>
#include "atb/atb_acl.h"
namespace atb {
namespace cinterfaceTest {
const int64_t MLAINOUTMLA = 12;
const int64_t MLAINOUTMLAPP = 28;
const int64_t MLAPREINOUT = 7;
const int64_t MLAPPREFILLINOUT = 9;
const int64_t blockSize = 128;
const int64_t numTokens = 32;
const int64_t numHeads = 32;
const int64_t kvHeads = 1;
const int64_t headSizeQk = 576;
const int64_t headSizeVo = 512;
const int64_t kSeqlen = 256;
const int64_t maskDim = 0;
const int64_t batch = numTokens * kSeqlen;
const int64_t numBlocks = 64;
const int64_t maxNumBlocksPerQuery = 16;
const int64_t maxSeqLen = 256;

const int64_t embeddimV = 128;
const int64_t qRopeSzie = 64;

const int64_t dims = 7168;
const int64_t dimB = 2112;
const int64_t dimC = 1536;
const int64_t dimD = 1;
const int64_t dimE = 125;
const int64_t dimF = 512;
const int64_t dimG = 2;

const int64_t sizeofFP16 = 2;
const int64_t quantScale0 = 0;
const int64_t quantOffset0 = 0;
const int64_t wdqkv = 1 * 224 * 2112 * 32;
const int64_t deScale0 = 2112;
const int64_t bias0 = 2112;

const int64_t gamma1 = 1536;
const int64_t beta1 = 1536;
const int64_t quantScale1 = 1;
const int64_t quantOffset1 = 1;
const int64_t wuq = 1 * 48 * numHeads * 192 * 32;
const int64_t deScale1 = numHeads * 192;
const int64_t bias1 = numHeads * 192;
const int64_t gamma2 = 512;
const int64_t cosNum = numTokens * 64;
const int64_t sinNum = numTokens * 64;
const int64_t wuk = numHeads * 128 * 512;


const int64_t kvCache = numBlocks * blockSize * 1 * 512;
const int64_t kvCacheC0 = numBlocks * blockSize * 1 * 576;
const int64_t kvCacheC2 = numBlocks * numHeads * 512 / 32 * blockSize * 32;
const int64_t kvCacheC3 = numBlocks * numHeads * 512 / 16 * blockSize * 16;

const int64_t kvCacheRope = numBlocks * blockSize * 1 * 64;
const int64_t kvCacheRopeC2 = numBlocks * numHeads * 64 / 16 * blockSize * 16;
const int64_t kvCacheRopeC3 = numBlocks * numHeads * 64 / 16 * blockSize * 16;
const int64_t slotmapping = numTokens;
const int64_t ctkvScale = 1;
const int64_t qNopeScale = numHeads;

const int64_t outTensor0C0 = numTokens * numHeads * 576;
const int64_t outTensor0C1 = numTokens * numHeads * 512;
const int64_t outTensor0C2 = numTokens * numHeads * 512;



const int64_t outTensor1C0 = numBlocks * blockSize * 576;
const int64_t outTensor1C1 = numBlocks * blockSize * 512;
const int64_t outTensor1C2 = numBlocks * numHeads * 512 / 32 * blockSize * 32;
const int64_t outTensor1C3 = numBlocks * numHeads * 512 / 16 * blockSize * 16;

const int64_t outTensor2 = numTokens * numHeads * 64;

const int64_t outTensor3C1 = numBlocks * blockSize * 1 * 64;
const int64_t outTensor3C2 = numBlocks * numHeads * 64 / 16 * blockSize * 16;


int64_t GetTensorSize(const aclTensor *input);
aclnnStatus Init(atb::Context **context, aclrtStream *stream, int64_t *deviceId);
aclnnStatus Destroy(atb::Context **context, aclrtStream *stream);
aclnnStatus CreateInOutData(size_t num, uint8_t **inoutHost, uint8_t **inoutDevice, size_t *inoutSize);
void CreateACLTensorInOut(const std::vector<int64_t> dims, aclDataType type, aclFormat format, aclTensor **list, size_t &i, void *inout);

//!
//! \brief 返回aclDataType对应的数据类型大小。
//!
//! \param dType 传入数据类型
//!
//! \return 返回整数值对应aclDataType大小
//!
uint64_t GetDataTypeSize(const aclDataType &dType);
}
}
#endif