/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_TILING_H
#define LCAL_TILING_H

#include <iostream>
#include <array>
#include "tiling_args.h"
#include "lcal_types.h"
#include "lcal_comm.h"
#include "lcoc.h"

namespace Lcal {
class CoCTilingFunc {
public:
    CoCTilingFunc(const CoCTilingFunc &) = delete;
    CoCTilingFunc &operator = (const CoCTilingFunc &) = delete;
    CoCTilingFunc() {}
    virtual ~CoCTilingFunc() {}
    CoCTilingData GenerateTiling(const TaskParam &taskParam, const CoCTiling &tiling);

    virtual bool CheckTiling(const TaskParam &taskParam);
    virtual void GetDefaultTiling(const TaskParam &taskParam);

protected:
    CoCTilingData cocTilingData = {};
};

class CoCMatmullReduceTilingFunc : public CoCTilingFunc {
public:
    CoCMatmullReduceTilingFunc(const CoCMatmullReduceTilingFunc &) = delete;
    CoCMatmullReduceTilingFunc &operator = (const CoCMatmullReduceTilingFunc &) = delete;
    CoCMatmullReduceTilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
}

class CoCMatmullReduceDeterTilingFunc : public CoCTilingFunc {
public:
    CoCMatmullReduceDeterTilingFunc(const CoCMatmullReduceDeterTilingFunc &) = delete;
    CoCMatmullReduceDeterTilingFunc &operator = (const CoCMatmullReduceDeterTilingFunc &) = delete;
    CoCMatmullReduceDeterTilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
}

class CoCAllgatherMatnulReduceScatterTilingFunc : public CoCTilingFunc {
public:
    CoCAllgatherMatnulReduceScatterTilingFunc(const CoCAllgatherMatnulReduceScatterTilingFunc &) = delete;
    CoCAllgatherMatnulReduceScatterTilingFunc &operator = (const CoCAllgatherMatnulReduceScatterTilingFunc &) = delete;
    CoCAllgatherMatnulReduceScatterTilingFunc() {}
    bool CheckTiling(const TaskParam &taskParam) override;
    void GetDefaultTiling(const TaskParam &taskParam) override;
};
}

#endif