/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/core/context_base.h"
#include <cmath>
#include <acl/acl.h>
#include "atb/core/tiling_buffer_pool/device_tiling_buffer_pool.h"
#include "atb/core/tiling_buffer_pool/host_tiling_buffer_pool.h"
#include "atb/utils/log.h"
#include "atb/types.h"
#include "atb/utils/config.h"
#include "atb/utils.h"
#include "atb/utils/probe.h"
#include "atb/utils/singleton.h"

namespace atb {
static constexpr size_t MAX_COPY_EVENT_NUM = 10;
static constexpr uint64_t TILING_BUFFER_BLOCK_SIZE = 1024 * 1024 * 3;
static constexpr uint32_t DEFAULT_EXECUTE_STREAM_NUMBER = 1;
thread_local ExecuteType ContextBase::executeType_ = EXECUTE_NORMAL;

ContextBase::ContextBase() {}

ContextBase::~ContextBase() noexcept
{
    try {
        DestoryCopyStreamAndEvents();
        if (Probe::IsOverflowCheck()) {
            FreeOverflowTensor();
        }
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << "An exception occurred when running DestoryCopyStreamAndEvents()."
                       << "Traceback: \n"
                       << e.what();
    }
}

Status ContextBase::Init()
{
    executeStreams_.resize(DEFAULT_EXECUTE_STREAM_NUMBER);

    hostTilingBufferPool_ = std::make_unique<HostTilingBufferPool>(GetSingleton<Config>().GetHostTilingBlockNum(),
                                                                   TILING_BUFFER_BLOCK_SIZE);
    if (!hostTilingBufferPool_) {
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    Status st = hostTilingBufferPool_->Init();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "ContextBase host tiling buffer pool init fail";
        return st;
    }

    deviceTilingBufferPool_ = std::make_unique<DeviceTilingBufferPool>(GetSingleton<Config>().GetDeviceTilingBlockNum(),
                                                                       TILING_BUFFER_BLOCK_SIZE);
    if (!deviceTilingBufferPool_) {
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    st = deviceTilingBufferPool_->Init();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "ContextBase device tiling buffer pool init fail";
        return st;
    }

    runnerPools_.resize(RUNNER_TYPE_MAX);
    if (Probe::IsOverflowCheck()) {
        st = CreateOverflowOutTensor();
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << "CreateOverflowOutTensor error!";
            return st;
        }
    }

    ATB_LOG(INFO) << "ContextBase init success";
    return NO_ERROR;
}

void ContextBase::Destroy()
{
    if (hostTilingBufferPool_) {
        hostTilingBufferPool_->Destroy();
    }

    if (deviceTilingBufferPool_) {
        deviceTilingBufferPool_->Destroy();
    }
}

Status ContextBase::SetExecuteStream(aclrtStream stream)
{
    executeStreams_.at(0) = stream;
    if (Probe::IsOverflowCheck()) {
        Status st = aclrtSetStreamOverflowSwitch(executeStreams_.at(0), 1);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << "aclrtSetStreamOverflowSwitch error! ret:" << st;
            return st;
        }
    }
    return NO_ERROR;
}

aclrtStream ContextBase::GetExecuteStream() const
{
    return executeStreams_.at(0);
}

Status ContextBase::SetAsyncTilingCopyStatus(bool status)
{
    const bool curStatus = (asyncTilingCopyStream_ != nullptr);
    if (status == curStatus) {
        ATB_LOG(INFO) << "ContextBase SetAsyncTilingCopyStatus do nothing";
        return NO_ERROR;
    }

    if (status) {
        ATB_LOG(INFO) << "ContextBase SetAsyncTilingCopyStatus create copy stream and events";
        return CreateCopyStreamAndEvents();
    } else {
        ATB_LOG(INFO) << "ContextBase SetAsyncTilingCopyStatus destroy copy stream and events";
        return DestoryCopyStreamAndEvents();
    }
}

bool ContextBase::GetAsyncTilingCopyStatus() const
{
    const bool curStatus = (asyncTilingCopyStream_ != nullptr);
    ATB_LOG(INFO) << "ContextBase GetAsyncTilingCopyStatus ret:" << curStatus;
    return curStatus;
}

aclrtStream ContextBase::GetAsyncTilingCopyStream() const
{
    return asyncTilingCopyStream_;
}

aclrtEvent ContextBase::GetAsyncTilingCopyEvent()
{
    aclrtEvent event = nullptr;
    if (asyncTilingCopyEvents_.empty()) {
        ATB_LOG(ERROR) << "ContextBase get sync tiling copy event fail, for asyncTilingCopyEvents is empty";
        return event;
    }

    event = asyncTilingCopyEvents_.at(asyncTilingCopyEventsIndex_);
    asyncTilingCopyEventsIndex_++;
    if (asyncTilingCopyEventsIndex_ == asyncTilingCopyEvents_.size()) {
        asyncTilingCopyEventsIndex_ = 0;
    }
    return event;
}

uint8_t *ContextBase::GetHostTilingBuffer()
{
    return hostTilingBufferPool_ ? hostTilingBufferPool_->GetBuffer() : nullptr;
}

uint8_t *ContextBase::GetDeviceTilingBuffer()
{
    return deviceTilingBufferPool_ ? deviceTilingBufferPool_->GetBuffer() : nullptr;
}

uint64_t ContextBase::GetTilingBufferBlockSize() const
{
    return TILING_BUFFER_BLOCK_SIZE;
}

Status ContextBase::CreateCopyStreamAndEvents()
{
    ATB_LOG(DEBUG) << "ContextBase aclrtCreateStream start";
    aclError ret = aclrtCreateStream(&asyncTilingCopyStream_);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << "ContextBase aclrtCreateStream fail, ret:" << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << "ContextBase aclrtCreateStream create success";

    ATB_LOG(INFO) << "ContextBase try create event:" << MAX_COPY_EVENT_NUM;
    asyncTilingCopyEvents_.resize(MAX_COPY_EVENT_NUM);
    for (size_t i = 0; i < asyncTilingCopyEvents_.size(); i++) {
        asyncTilingCopyEvents_.at(i) = nullptr;
        ret = aclrtCreateEvent(&asyncTilingCopyEvents_.at(i));
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << "ContextBase aclrtCreateEvent ret:" << ret;
            return ERROR_CANN_ERROR;
        }
        ATB_LOG(INFO) << "ContextBase aclrtCreateEvent create success";
    }

    return NO_ERROR;
}

Status ContextBase::DestoryCopyStreamAndEvents()
{
    aclError ret = ACL_SUCCESS;
    if (asyncTilingCopyStream_) {
        ret = aclrtSynchronizeStream(asyncTilingCopyStream_);
        ATB_LOG_IF(ret != ACL_SUCCESS, ERROR) << "aclrtSynchronizeStream fail, ret:" << ret;
        ret = aclrtDestroyStream(asyncTilingCopyStream_);
        ATB_LOG_IF(ret != ACL_SUCCESS, ERROR) << "aclrtDestroyStream fail, ret:" << ret;
    }
    for (size_t i = 0; i < asyncTilingCopyEvents_.size(); i++) {
        if (asyncTilingCopyEvents_.at(i)) {
            ret = aclrtSynchronizeEvent(asyncTilingCopyEvents_.at(i));
            ATB_LOG_IF(ret != ACL_SUCCESS, ERROR) << "aclrtSynchronizeEvent fail, ret:" << ret;
            ret = aclrtDestroyEvent(asyncTilingCopyEvents_.at(i));
            ATB_LOG_IF(ret != ACL_SUCCESS, ERROR) << "aclrtDestroyEvent fail, ret:" << ret;
            asyncTilingCopyEvents_.at(i) = nullptr;
        }
    }

    return NO_ERROR;
}

RunnerPool &ContextBase::GetRunnerPool(RunnerType runnerType)
{
    return runnerPools_.at(runnerType);
}

const Tensor &ContextBase::GetOverflowKernelOutTensor()
{
    return overflowOutTensor_;
}

Status ContextBase::CreateOverflowOutTensor()
{
    if (overflowOutTensor_.deviceData != nullptr) {
        return NO_ERROR;
    }
    overflowOutTensor_.desc.dtype = ACL_INT32;
    overflowOutTensor_.desc.format = ACL_FORMAT_ND;
    overflowOutTensor_.desc.shape.dimNum = 1;
    overflowOutTensor_.desc.shape.dims[0] = 1;
    overflowOutTensor_.dataSize = Utils::GetTensorSize(overflowOutTensor_.desc);
    int ret = aclrtMalloc(&overflowOutTensor_.deviceData, overflowOutTensor_.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "overflowOutTensor malloc failed!";
        return ERROR_OUT_OF_DEVICE_MEMORY;
    }

    return NO_ERROR;
}

Status ContextBase::FreeOverflowTensor()
{
    if (overflowOutTensor_.deviceData != nullptr) {
        Status st = aclrtFree(overflowOutTensor_.deviceData);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << "aclrtFree fail, ret:" << st;
            return st;
        }
    }
    return NO_ERROR;
}

Status ContextBase::SetExecuteStreams(const std::vector<aclrtStream> &streams)
{
    if (streams.size() < 1) {
        ATB_LOG(ERROR) << "SetExecuteStreams failed! size of streams should be bigger than 0.";
        return ERROR_INVALID_PARAM;
    }
    executeStreams_ = streams;
    return NO_ERROR;
}

std::vector<aclrtStream> ContextBase::GetExecuteStreams()
{
    return executeStreams_;
}

Status ContextBase::SetExecuteType(ExecuteType type)
{
    if (type != EXECUTE_NORMAL && type != EXECUTE_PRELAUNCH && type != EXECUTE_LAUNCH) {
        ATB_LOG(ERROR)
            << "SetExecuteType failed! executeType should be EXECUTE_NORMAL, EXECUTE_LAUNCH or EXECUTE_PRELAUNCH."
            << " now executeType is:" << type;
        return ERROR_INVALID_PARAM;
    }
    executeType_ = type;
    return NO_ERROR;
}

ExecuteType ContextBase::GetExecuteType()
{
    return executeType_;
}

Status ContextBase::SetL2CacheBuffer(const void *l2CacheBuffer, size_t bufferSize)
{
    if (!l2CacheBuffer) {
        ATB_LOG(INFO) << "SetL2CacheBuffer failed! l2CacheBuffer is nullptr";
        return ERROR_INVALID_PARAM;
    }
    l2CacheBuffer_ = l2CacheBuffer;
    l2CacheBufferSize_ = bufferSize;
    return NO_ERROR;
}

void *GetL2CacheBuffer() const
{
    return l2CacheBuffer_;
}

size_t GetL2CacheBufferSize() const
{
    return l2CacheBufferSize_;
}

} // namespace atb
