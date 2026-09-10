/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ACL_OP_EXECUTOR_H
#define ATB_ACL_OP_EXECUTOR_H

#include <memory>
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

namespace atb {
class atbAclOpExecutor {
public:
    explicit atbAclOpExecutor(aclOpExecutor *executor);
    ~atbAclOpExecutor();

    atbAclOpExecutor(const atbAclOpExecutor &) = delete;
    atbAclOpExecutor &operator=(const atbAclOpExecutor &) = delete;

    aclOpExecutor *Get() const;
    bool IsRepeatable() const;

private:
    aclOpExecutor *executor_ = nullptr;
    bool repeatable_ = false;
};
} // namespace atb

#endif // ATB_ACL_OP_EXECUTOR_H
