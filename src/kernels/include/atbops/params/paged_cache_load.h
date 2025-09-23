/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATBOPS_PARAMS_PAGED_CACHE_LOAD_H
#define ATBOPS_PARAMS_PAGED_CACHE_LOAD_H

#include <vector>

namespace AtbOps {
namespace OpParam {
struct PagedCacheLoad {
    enum class Type {
        PAGED_CACHE_LOAD_ND = 0,
        PAGED_CACHE_LOAD_NZ = 1,
    };
    Type type = Type::PAGED_CACHE_LOAD_ND;
    bool cuSeqLens = false;
    bool hasSeqStarts = false;

    bool operator==(const PagedCacheLoad &other) const
    {
        return this->type == other.type;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_PAGED_CACHE_LOAD_H