/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file auto_tiling_register.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_AUTO_TILING_REGISTER_H_
#define OPS_BUILT_IN_OP_TILING_AUTO_TILING_REGISTER_H_

#include "vector_tiling_rt2.h"

#include <vector>

using AutoTilingFunc = bool (*)(gert::TilingContext *, const optiling::OpInfoImpl *);
using AutoTilingParseFunc = optiling::AutoTilingCompileInfo *(*)(const char *op_type,
                                                                 const nlohmann::json &json_compile_info);

constexpr size_t PATTERN_BASE = 0x10;
constexpr size_t PATTERN_SIZE = static_cast<size_t>(optiling::SchPattern::DEFAULT) - PATTERN_BASE;

inline size_t PatternIndex(optiling::SchPattern _pattern)
{
    return static_cast<size_t>(_pattern) - PATTERN_BASE;
}

/*
 * Isolate AutoTiling registry from system liboptiling (STB_GNU_UNIQUE merge).
 * Out-of-line RegisterTiling/Parser live in stubs/.../auto_tiling_register.cpp.
 * CMake -include this header for canndev tiling TUs so same-dir real header is skipped.
 */
class AsdOpsAutoTilingRegister {
public:
    AsdOpsAutoTilingRegister(optiling::SchPattern _pattern, AutoTilingFunc _tiling_func, AutoTilingParseFunc _parser)
    {
        size_t index = PatternIndex(_pattern);
        auto &register_parser = RegisterParser();
        register_parser[index] = _parser;
        auto &register_tiling = RegisterTiling();
        register_tiling[index] = _tiling_func;
    };
    ~AsdOpsAutoTilingRegister() = default;
    static std::array<AutoTilingParseFunc, PATTERN_SIZE> &RegisterParser();
    static std::array<AutoTilingFunc, PATTERN_SIZE> &RegisterTiling();
};

#define AutoTilingRegister AsdOpsAutoTilingRegister

#define REGISTER_AUTO_TILING(pattern, tilingfunc, parsefunc)                                                           \
    AutoTilingRegister __attribute__((unused)) g_auto_tiling_register_##tilingfunc(pattern, tilingfunc, parsefunc)

#endif // OPS_BUILT_IN_OP_TILING_AUTO_TILING_REGISTER_H_
