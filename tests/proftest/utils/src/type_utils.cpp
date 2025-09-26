/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "type_utils.h"
#include <cstdint>
#include <cstring>
#include <cmath>
float16 FloatToFloat16(float fp32)
{
    if (fp32 == 0.0f) {
        return (std::signbit(fp32) ? 0x8000 : 0x0000);
    }

    uint32_t float_bits;
    static_assert(sizeof(float) == sizeof(uint32_t), "Float size mismatch");
    std::memcpy(&float_bits, &fp32, sizeof(float));

    const uint32_t sign = (float_bits >> 31) & 0x1;
    const uint32_t exp = (float_bits >> 23) & 0xFF;
    const uint32_t mant = float_bits & 0x7FFFFF;
    if (exp == 0xFF) {
        if (mant == 0) {
            return (sign << 15) | 0x7C00;
        } else {
            return (sign << 15) | 0x7C00 | (mant >> 13);
        }
    }

    int32_t exp_fp16 = static_cast<int32_t>(exp) - 127 + 15;
    if (exp_fp16 <= 0) {
        return (sign << 15);
    }

    if (exp_fp16 >= 0x1F) {
        return (sign < 15) | 0x7C00;
    }

    uint32_t mant24 = (1 << 23) | mant;
    uint32_t round_bits = mant24 & 0x1FFF;
    uint32_t base = (mant24 >> 13) & 0x3FF;

    if (round_bits > 0x1000 || (round_bits == 0x1000 && (base & 1))) {
        base++;
        if (base > 0xFF) {
            base = 0;
            exp_fp16++;
            if (exp_fp16 >= 0x1F) {
                return (sign << 15) | 0x7C00;
            }
        }
    }

    return (sign << 15) | (exp_fp16 << 10) | base;
}

bfloat16 FloatToBfloat16(float fp32)
{
    if (fp32 == 0.0f) {
        return (std::signbit(fp32) ? 0x8000 : 0x0000);
    }

    uint32_t float_bits;
    static_assert(sizeof(float) == sizeof(uint32_t), "Float size mismatch");
    std::memcpy(&float_bits, &fp32, sizeof(float));

    bfloat16 bfloat16_bits = static_cast<bfloat16>(float_bits >> 16);

    const uint32_t exp = (float_bits >> 23) & 0xFF;
    const uint32_t mant = float_bits & 0x7FFFFF;
    if (exp == 0xFF && mant != 0) {
        bfloat16_bits |= 0x01;
    }

    return bfloat16_bits;
}

float Float16ToFloat(float16 fp16)
{
    const uint32_t sign = (fp16 >> 15) & 0x1;
    const uint32_t exp_f16 = (fp16 >> 10) & 0x1F;
    const uint32_t mant_f16 = fp16 & 0x3FF;
    if (exp_f16 == 0x1F) {
        uint32_t inf_nan = (sign << 31) | 0x7F800000 | (mant_f16 << 13);
        float result;
        memcpy(&result, &inf_nan, sizeof(float));
        return result;
    }
    if (exp_f16 == 0) {
        if (mant_f16 == 0) {
            uint32_t sign_bit = sign << 31;
            float result;
            memcpy(&result, &sign_bit, sizeof(float));
            return result;
        } else {
            uint32_t shift = 0;
            uint32_t mant = mant_f16;
            while ((mant & 0x4000) == 0) {
                mant <<= 1;
                ++shift;
            }
            const int32_t exp_float = -14 - shift + 1 + 127;
            const uint32_t mant_float = (mant & 0x3FF) << 13;
            uint32_t combined = (sign << 31) | (exp_float << 23) | mant_float;
            float result;
            memcpy(&result, &combined, sizeof(float));
            return result;
        }
    }

    const uint32_t exp_float = exp_f16 + 112;
    const uint32_t mant_float = mant_f16 << 13;
    uint32_t combined = (sign << 31) | (exp_float << 23) | mant_float;
    float result;
    memcpy(&result, &combined, sizeof(float));
    return result;
}

float Bfloat16ToFloat(bfloat16 bf16)
{
    uint32_t float_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    memcpy(&result, &float_bits, sizeof(float));
    return result;
}