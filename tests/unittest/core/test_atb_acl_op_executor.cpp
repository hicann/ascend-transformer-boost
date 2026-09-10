/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <type_traits>
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <cpp-stub/src/stub.h>
#include "atb/utils/atb_acl_op_executor.h"

using namespace atb;

namespace {
// atbAclOpExecutor 必须不可拷贝、不可赋值，否则会破坏 RAII 语义（重复析构同一 executor）
static_assert(!std::is_copy_constructible<atbAclOpExecutor>::value, "atbAclOpExecutor must not be copy constructible");
static_assert(!std::is_copy_assignable<atbAclOpExecutor>::value, "atbAclOpExecutor must not be copy assignable");

// aclSetAclOpExecutorRepeatable 的 stub 返回值
aclnnStatus g_setRepeatableRet = ACL_SUCCESS;
// aclDestroyAclOpExecutor 是否被调用
bool g_destroyCalled = false;

aclnnStatus MockAclSetAclOpExecutorRepeatable(aclOpExecutor *executor)
{
    return g_setRepeatableRet;
}

aclnnStatus MockAclDestroyAclOpExecutor(aclOpExecutor *executor)
{
    g_destroyCalled = true;
    return ACL_SUCCESS;
}
} // namespace

// 纯逻辑单测：ACL 函数已被 stub，不依赖硬件/平台，所有 soc 均可运行
class TestAclOpExecutor : public testing::Test {
protected:
    void SetUp() override
    {
        g_setRepeatableRet = ACL_SUCCESS;
        g_destroyCalled = false;
    }
};

// 用例1：set repeatable 成功（ret=0）→ repeatable_=true，Get 返回原指针
TEST_F(TestAclOpExecutor, ConstructRepeatableTrue)
{
    Stub stub;
    g_setRepeatableRet = ACL_SUCCESS;
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclSetAclOpExecutorRepeatable, MockAclSetAclOpExecutorRepeatable);
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclDestroyAclOpExecutor, MockAclDestroyAclOpExecutor);

    aclOpExecutor *raw = reinterpret_cast<aclOpExecutor *>(0x1234);
    atbAclOpExecutor executor(raw);
    EXPECT_TRUE(executor.IsRepeatable());
    EXPECT_EQ(executor.Get(), raw);
}

// 用例2：set repeatable 失败（ret≠0）→ repeatable_=false
TEST_F(TestAclOpExecutor, ConstructRepeatableFalse)
{
    Stub stub;
    g_setRepeatableRet = 561000; // 非 ACL_SUCCESS，模拟不可复用
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclSetAclOpExecutorRepeatable, MockAclSetAclOpExecutorRepeatable);
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclDestroyAclOpExecutor, MockAclDestroyAclOpExecutor);

    aclOpExecutor *raw = reinterpret_cast<aclOpExecutor *>(0x1234);
    atbAclOpExecutor executor(raw);
    EXPECT_FALSE(executor.IsRepeatable());
    EXPECT_EQ(executor.Get(), raw);
}

// 用例3：repeatable=true 的 executor 析构时，应调用 aclDestroyAclOpExecutor
TEST_F(TestAclOpExecutor, DestructRepeatableCallsDestroy)
{
    Stub stub;
    g_setRepeatableRet = ACL_SUCCESS;
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclSetAclOpExecutorRepeatable, MockAclSetAclOpExecutorRepeatable);
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclDestroyAclOpExecutor, MockAclDestroyAclOpExecutor);

    g_destroyCalled = false;
    {
        aclOpExecutor *raw = reinterpret_cast<aclOpExecutor *>(0x1234);
        atbAclOpExecutor executor(raw);
        EXPECT_TRUE(executor.IsRepeatable());
    }
    EXPECT_TRUE(g_destroyCalled);
}

// 用例4：repeatable=false 的 executor 析构时，不应调用 aclDestroyAclOpExecutor（避免 double free）
TEST_F(TestAclOpExecutor, DestructNotRepeatableSkipsDestroy)
{
    Stub stub;
    g_setRepeatableRet = 561000;
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclSetAclOpExecutorRepeatable, MockAclSetAclOpExecutorRepeatable);
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclDestroyAclOpExecutor, MockAclDestroyAclOpExecutor);

    g_destroyCalled = false;
    {
        aclOpExecutor *raw = reinterpret_cast<aclOpExecutor *>(0x1234);
        atbAclOpExecutor executor(raw);
        EXPECT_FALSE(executor.IsRepeatable());
    }
    EXPECT_FALSE(g_destroyCalled);
}

// 用例5：nullptr 构造，不调用 set repeatable，Get 返回 nullptr，析构安全
TEST_F(TestAclOpExecutor, ConstructNullSafe)
{
    Stub stub;
    g_setRepeatableRet = ACL_SUCCESS;
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclSetAclOpExecutorRepeatable, MockAclSetAclOpExecutorRepeatable);
    stub.set((aclnnStatus(*)(aclOpExecutor *))aclDestroyAclOpExecutor, MockAclDestroyAclOpExecutor);

    g_destroyCalled = false;
    atbAclOpExecutor executor(nullptr);
    EXPECT_EQ(executor.Get(), nullptr);
    EXPECT_FALSE(executor.IsRepeatable());
    EXPECT_FALSE(g_destroyCalled);
}
