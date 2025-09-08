/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "atb/core/aclnn/executor_manager.h"
#include "atb/utils/log.h"

namespace atb {
int ExecutorManager::IncreaseReference(aclOpExecutor *executor)
{
    std::map<aclOpExecutor *, int>::iterator it = this->executorCount_.find(executor);
    if (it == this->executorCount_.end()) {
        ATB_LOG(INFO) << "ATB aclnn Op Cache: Executor addr[" << executor << "] not found in ExecutorManager, add one";
        this->executorCount_[executor] = 1;
        return 1;
    }

    int &count = it->second;
    count += 1;
    ATB_LOG(INFO) << "ATB aclnn Op Cache: ExecutorManager Executor addr[" << executor << "] increase reference to "
                  << count;
    return count;
}

int ExecutorManager::DecreaseReference(aclOpExecutor *executor)
{
    std::map<aclOpExecutor *, int>::iterator it = this->executorCount_.find(executor);
    if (it == this->executorCount_.end()) {
        ATB_LOG(ERROR) << "ATB aclnn Op Cache: Executor addr[" << executor << "] not found in ExecutorManager";
        return 0;
    }
    int &count = it->second;
    if (count == 1) {
        ATB_LOG(INFO) << "ATB aclnn Op Cache: delete Executor addr[" << executor << "]";
        this->executorCount_.erase(executor);
        return 0;
    }

    count -= 1;
    ATB_LOG(INFO) << "ATB aclnn Op Cache: ExecutorManager Executor addr[" << executor << "] decrease reference to "
                  << count;
    return count;
}

std::string ExecutorManager::PrintExecutorCount()
{
    std::stringstream ss;
    ss << "ATB aclnn Op Cache: Executor Summary ";
    std::map<aclOpExecutor *, int>::iterator it;
    for (it = this->executorCount_.begin(); it != this->executorCount_.end(); it++) {
        ss << "Executor Addr[" << it->first << "] count " << it->second << " ";
    }
    return ss.str();
}

} // namespace atb
