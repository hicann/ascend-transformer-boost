/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lcal_internal.h"
#include <map>
#include <mutex>
#include <vector>
#include <mki/utils/log/log.h>
#include <mki/utils/env/env.h>
#include <runtime/kernel.h>
#include "ccl_kernel_args.h"
#include "coc_kernel_args.h"
#include "lcoc.h"

using namespace std;
using namespace Mki;

extern const int LCAL_CCE_BIN_STR[];
asm(R"(.section .rodata, "a", @progbits
LCAL_CCE_BIN_STR:.incbin "/tmp/lcal_cce.o"
.byte 0
.previous)");

constexpr int LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246;
constexpr int COC_RT_DEV_BINARY_MAGIC_ELF = 0x43554245;

namespace Lcal {
const std::map<HcclDataType, std::string> DATATYPE2NAME = {
    { HCCL_DATA_TYPE_INT32, "int" },
    { HCCL_DATA_TYPE_INT16, "int16_t" },
    { HCCL_DATA_TYPE_INT8, "int8_t" },
    { HCCL_DATA_TYPE_INT64, "int64_t" },
    { HCCL_DATA_TYPE_FP32, "float" },
    { HCCL_DATA_TYPE_FP16, "float16_t" },
    { HCCL_DATA_TYPE_BFP16, "bfloat16_t" }
};

template<class T>
int RegisterBinaryKernel(const string &funcName, int8_t *funSig, const T *binStrPtr, int magic, int len = 0)
{
    rtDevBinary_t binary;
    void *binHandle = nullptr;
    binary.data = binStrPtr;
    binary.length = (len == 0 ? LCAL_1OP_BIN_SIZE : len);

    binary.magic = magic;
    binary.version = 0;
    rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
    if (rtRet != RT_ERROR_NONE) {
        MKI_LOG(WARN) << "rtDevBinaryRegister failed! " << to_string(rtRet) << ", funcName = " << funcName;
        return LCAL_ERROR_INTERNAL;
    }
    rtRet = rtFunctionRegister(binHandle, funSig, funcName.c_str(), funcName.c_str(), 0);
    if (rtRet != RT_ERROR_NONE) {
        MKI_LOG(WARN) << "rtFunctionRegister failed! " << to_string(rtRet) << ", funcName = " << funcName;
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int8_t *GetFunSig(LcalType type, HcclDataType dataType, uint64_t devType = 0)
{
    constexpr int sigOffset = 16;
    constexpr int sigSkew = 0x1000;
    return reinterpret_cast<int8_t *>((static_cast<uint64_t>(type) << sigOffset << sigOffset) +
        (static_cast<uint64_t>(dataType)<< sigOffset) + devType + sigSkew);
}

const int* FindNextOpStart(const int opStartMaigc, const int* cclBinEndPtr, const int* cclBinPtr)
{
    if (cclBinPtr == nullptr) {
        MKI_LOG(ERROR) << "FindNextOpStart failed! cclBinPtr is nullptr";
        return nullptr;
    }
    while (*cclBinPtr != opStartMaigc and cclBinPtr < cclBinEndPtr) {
        cclBinPtr++;
    }
    if (*cclBinPtr == opStartMaigc) {
        cclBinPtr++;
    }
    return cclBinPtr;
}

int RegistCCLOp2Kernel(const int* cclBinPtr, const int* nextPtr)
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT8,
                                           HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16,
                                           HCCL_DATA_TYPE_INT64 };
    std::vector<LcalType> registerCCLTypesOp2 = {
        LcalType::ALL_GATHER, LcalType::REDUCE_SCATTER, LcalType::ALL2ALL,
    };
    int res = LCAL_SUCCESS;
    for (auto ccl : registerCCLTypesOp2) {
        for (auto t : registerTypes) {
            res = RegisterBinaryKernel(LCAL_TYPE2NAME.at(ccl) + "_" + DATATYPE2NAME.at(t), GetFunSig(ccl, t),
                cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC, (nextPtr - cclBinPtr) * sizeof(int));
        }
    }
    if (res != LCAL_SUCCESS) {
        return res;
    }
    res = RegisterBinaryKernel(LCAL_TYPE2NAME.at(LcalType::BROADCAST),
        GetFunSig(LcalType::BROADCAST, HCCL_DATA_TYPE_RESERVED), cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC);
    return res;
}

int RegistCCLOp1Kernel(const int* cclBinPtr, const int* nextPtr)
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT8,
                                           HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16,
                                           HCCL_DATA_TYPE_INT64 };
    std::vector<LcalType> registerCCLTypesOp1 = {
        LcalType::ALL_REDUCE,
    };
    int res = LCAL_SUCCESS;
    for (auto ccl : registerCCLTypesOp1) {
        for (auto t : registerTypes) {
            res = RegisterBinaryKernel(LCAL_TYPE2NAME.at(ccl) + "_" + DATATYPE2NAME.at(t), GetFunSig(ccl, t),
                cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC, (nextPtr - cclBinPtr) * sizeof(int));
        }
    }
    return res;
}

int RegistCCLKernel(const int32_t opGroup)
{
    const int* cclBinStr = LCAL_CCE_BIN_STR;
    auto cclBinEndPtr = cclBinStr + LCAL_1OP_BIN_SIZE / sizeof(int);
    const int* cclBinPtr = cclBinStr + 1;
    constexpr int opStartMaigc = 0x44444444;
    const int* nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, cclBinPtr);
    if (nextPtr == nullptr) {
        return LCAL_ERROR_INTERNAL;
    }

    constexpr int32_t smallGroupNum = 2;

    for (int32_t opGroupIdx = 0; opGroupIdx < opGroup; ++opGroupIdx) {
        for (int32_t opIdx = 0; opIdx < smallGroupNum; ++opIdx) {
            cclBinPtr = nextPtr;
            nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, nextPtr);
            if (cclBinPtr == nullptr || cclBinPtr == cclBinEndPtr || nextPtr == nullptr) {
                return LCAL_ERROR_INTERNAL;
            }
        }
    }

    int ret = 0;
    ret = RegistCCLOp1Kernel(cclBinPtr, nextPtr);
    if (ret != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }

    cclBinPtr = nextPtr;
    nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, nextPtr);
    if (cclBinPtr == nullptr || cclBinPtr == cclBinEndPtr || nextPtr == nullptr) {
        return LCAL_ERROR_INTERNAL;
    }

    ret = RegistCCLOp2Kernel(cclBinPtr, nextPtr);
    if (ret != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

void RegistCoCKernel()
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16 };
    vector<vector<LcalType>> registerCOCTypes = {
        { LcalType::MATMUL_ALL_REDUCE },
        { LcalType::ALL_GATHER_MATMUL_REDUCE_SCATTER},
    };

    auto cocCceBinStr = LCAL_CCE_BIN_STR + LCAL_1OP_BIN_SIZE / sizeof(int);
    for (auto lcalTypeGroup : registerCOCTypes) {
        for (auto lcalType : lcalTypeGroup) {
            for (auto t : registerTypes) {
                RegisterBinaryKernel(LCAL_TYPE2NAME.at(lcalType) + "_" + DATATYPE2NAME.at(t), GetFunSig(lcalType, t),
                    cocCceBinStr, COC_RT_DEV_BINARY_MAGIC_ELF);
            }
        }
        cocCceBinStr += LCAL_1OP_BIN_SIZE / sizeof(int);
    }
}

int RegistKernel(const int32_t opGroup)
{
    static bool init = false;
    static mutex mut;
    lock_guard<mutex> guard(mut);
    if (init) {
        return 0;
    }
    RegistCoCKernel();
    RegistCCLKernel(opGroup);
    init = true;
    return LCAL_SUCCESS;
}

int64_t Count2Size(int64_t count, const HcclDataType &dataType)
{
    int64_t dataSize = LCAL_INVALID_VALUE;
    if (dataType == HCCL_DATA_TYPE_INT8 || dataType == HCCL_DATA_TYPE_UINT8) {
        dataSize = count;
    } else if (dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_FP16 ||
               dataType == HCCL_DATA_TYPE_BFP16 || dataType == HCCL_DATA_TYPE_UINT16) {
        dataSize = count * sizeof(int16_t);
    } else if (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_INT32 ||
               dataType == HCCL_DATA_TYPE_UINT32) {
        dataSize = count * sizeof(int32_t);
    } else if (dataType == HCCL_DATA_TYPE_INT64 || dataType == HCCL_DATA_TYPE_UINT64) {
        dataSize = count * sizeof(int64_t);
    } else {
        MKI_LOG(ERROR) << "unknown datatype";
    }
    return dataSize;
}


}