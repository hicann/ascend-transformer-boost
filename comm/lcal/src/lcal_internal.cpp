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

namespcae Lcal {
const std::map<HcclDataType, std::string> DATATYPE2NAME = {
    {HCCL_DATA_TYPE_INT32, "int"},
    {HCCL_DATA_TYPE_INT16, "int16_t"},
    {HCCL_DATA_TYPE_INT8, "int8_t"},
    {HCCL_DATA_TYPE_INT64, "int64_t"},
    {HCCL_DATA_TYPE_FP32, "float"},
    {HCCL_DATA_TYPE_FP16, "float16_t"},
    {HCCL_DATA_TYPE_BFP16, "bfloat16_t"},
};


template<class T>
int RegisterBinaryKernel()



}