/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_ALLOCATOR_H
#define ATB_ALLOCATOR_H
#include "atb/types.h"

//!
//! \file allocator.h
//!
//! \brief 定义加速库内存管理类
//!

//!
//! \namespace atb
//!
//! \brief 加速库的命名空间.
//!
namespace atb {

//!
//! \class Allocator.
//!
//! \brief 加速库Allocator类.
//!
//! 该接口类定义了加速库内存管理类，用于分配整图下发模式下的Device侧内存
//!
class Allocator {
public:
    //! \brief 默认构造函数.
    Allocator() = default;

    //! \brief 默认析构函数.
    virtual ~Allocator() = default;

    //!
    //! \brief 分配一块bufferSize大小的Device侧内存，返回分配好的内存首地址
    //!
    //! \param bufferSize 需要申请的内存大小
    //!
    //! \return 返回申请出来的指针
    //!
    virtual void* Allocate(size_t bufferSize) = 0;

    //!
    //! \brief 回收已经分配过的Device侧内存
    //!
    //! \param addr 需要释放的内存地址
    //!
    //! \return 状态值.如果释放成功，返回NO_ERROR
    //!
    virtual Status Deallocate(void *addr) = 0;
};
} // namespace atb
#endif