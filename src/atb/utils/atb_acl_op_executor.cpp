/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This software is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "atb/utils/atb_acl_op_executor.h"
#include "atb/utils/log.h"

namespace atb {
atbAclOpExecutor::atbAclOpExecutor(aclOpExecutor *executor) : executor_(executor)
{
    if (executor_ != nullptr) {
        aclnnStatus ret = aclSetAclOpExecutorRepeatable(executor_);
        repeatable_ = (ret == ACL_SUCCESS);
        ATB_LOG(INFO) << "atbAclOpExecutor construct, set repeatable return: " << ret
                      << ", repeatable: " << (ret == ACL_SUCCESS);
    }
}

atbAclOpExecutor::~atbAclOpExecutor()
{
    if (executor_ != nullptr) {
        if (repeatable_) {
            aclnnStatus ret = aclDestroyAclOpExecutor(executor_);
            ATB_LOG(INFO) << "atbAclOpExecutor destruct, destroy executor return: " << ret;
        }
        executor_ = nullptr;
    }
}

aclOpExecutor *atbAclOpExecutor::Get() const
{
    return executor_;
}

bool atbAclOpExecutor::IsRepeatable() const
{
    return repeatable_;
}
} // namespace atb
