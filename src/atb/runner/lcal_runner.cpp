/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/lcal_runner.h"
#include <atb/utils/log.h>
#include <acl/acl.h>
#include "atb/utils/config.h"
#include "atb/core/comm_pool.h"
#include "atb/utils/singleton.h"

namespace atb {
LcalRunner::LcalRunner(const std::string &name, RunnerType runnerType, int32_t rank, int32_t rankSize,
                       const infer::CommMode commMode, const std::string &commDomain)
    : Runner(name), runnerType_(runnerType), rank_(rank), rankSize_(rankSize), commMode_(commMode),
      commDomain_(commDomain)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LcalRunner::LcalRunner " << runnerType_ << " called, rank : " << rank << "/"
                  << rankSize << " commMode: " << commMode_ << " commDomain: " <<commDomain_;
    InitLcalComm();
}

LcalRunner::~LcalRunner() {}

Lcal::LcalComm *LcalRunner::GetLcalComm()
{
    return lcalComm_.get();
}

void LcalRunner::InitLcalComm()
{
    lcalComm_ = GetSingleton<CommPool<Lcal::LcalComm>>().GetComm(rank_, commDomain_,
                                                                 std::bind(&LcalRunner::CreateLcalComm, this));
    if (lcalComm_) {
        ATB_LOG(INFO) << GetLogPrefix() << "get lcal comm from comm pool success, rank : " << rank_;
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "get lcal comm from comm pool failed, rank : " << rank_;
    }
}

std::shared_ptr<Lcal::LcalComm> LcalRunner::CreateLcalComm()
{
    int32_t curDevId = -1;
    int ret = aclrtGetDevice(&curDevId);
    if (ret != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclrtGetDevice fail! rank: " << rank_;
        return std::shared_ptr<Lcal::LcalComm>();
    }
    std::shared_ptr<Lcal::LcalComm> newLcalComm = std::make_shared<Lcal::LcalComm>(rank_, rankSize_, curDevId);
    if (!newLcalComm) {
        ATB_LOG(ERROR) << GetLogPrefix() << "new Lcal Comm of rank fail: " << rank_;
        return std::shared_ptr<Lcal::LcalComm>();
    }
    if (commMode_ == infer::CommMode::COMM_MULTI_PROCESS) {
        lcalErrorCode_ = newLcalComm->Init();
    } else if (commMode_ == infer::CommMode::COMM_MULTI_THREAD) {
        lcalErrorCode_ = newLcalComm->InitThread(commDomain_);
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "Invalid commMode: " << commMode_;
        return std::shared_ptr<Lcal::LcalComm>();
    }
    if (lcalErrorCode_ != Lcal::LCAL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "init LcalComm of rank: " << rank_ << " failed, ret: " << ret;
        return std::shared_ptr<Lcal::LcalComm>();
    }
    return newLcalComm;
}

Status LcalRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    (void)runnerVariantPack;
    if (lcalErrorCode_ == Lcal::LCAL_SUCCESS) {
        return NO_ERROR;
    }
    if (lcalErrorCode_ == Lcal::OUT_OF_DEVICE_MEMORY) {
        ATB_LOG(ERROR) << "error code:" << ERROR_OUT_OF_DEVICE_MEMORY
                       << ", out of NPU memory! Please check if the memory is enough.";
        return ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return ERROR_INTERNAL_ERROR;
}
} // namespace atb
