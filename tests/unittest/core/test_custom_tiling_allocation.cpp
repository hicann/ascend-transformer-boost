/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <atb/operation.h>
#include "atb/core/allocator/default_device_allocator.h"
using namespace atb;

TEST(TestCustomAllocatorTiling, TestCustomCase)
{
    DefaultDeviceAllocator defaultDeviceAllocator;
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream exeStream = nullptr;
    aclrtCreateStream(&exeStream);
    atb::Context *context = nullptr;
    Status st = atb::CreateContext(&context, [&](size_t size) { return defaultDeviceAllocator.Allocate(size); }, [&](void * addr) { defaultDeviceAllocator.Deallocate(addr); });
    EXPECT_EQ(st, atb::NO_ERROR);
    st = atb::DestroyContext(context);
    EXPECT_EQ(st, atb::NO_ERROR);
}

TEST(TestCustomAllocatorTiling, TestOriginCase)
{
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream exeStream = nullptr;
    aclrtCreateStream(&exeStream);
    atb::Context *context = nullptr;
    Status st = atb::CreateContext(&context);
    EXPECT_EQ(st, atb::NO_ERROR);
    st = atb::DestroyContext(context);
    EXPECT_EQ(st, atb::NO_ERROR);
}