/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "acltransformer/context/work_context.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "buffer_device.h"
#include "buffer_host.h"

namespace AclTransformer {
WorkContext::WorkContext()
{
    uint64_t bufferRing = GetHostTilingBufferRing();
    uint64_t bufferSize = GetHostTilingBufferSize();
    ASD_LOG(FATAL) << "hosttiling bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    hostTilingBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        hostTilingBuffers_.at(i).reset(new BufferHost(bufferSize));
    }

    bufferRing = GetTilingBufferRing();
    bufferSize = GetTilingBufferSize();
    ASD_LOG(FATAL) << "tiling bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    tilingBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        tilingBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }

    bufferRing = GetWorkspaceBufferRing();
    bufferSize = GetWorkspaceBufferSize();
    ASD_LOG(FATAL) << "workspace bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    workspaceBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        workspaceBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }

    bufferRing = GetIntermediateBufferRing();
    bufferSize = GetIntermediateBufferSize();
    ASD_LOG(FATAL) << "intermediate bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    intermediateBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        intermediateBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }
}

WorkContext::~WorkContext() {}

void *WorkContext::GetHostTilingBuffer(uint64_t bufferSize)
{
    if (hostTilingBufferOffset_ == hostTilingBuffers_.size()) {
        hostTilingBufferOffset_ = 0;
    }
    return hostTilingBuffers_.at(hostTilingBufferOffset_++)->GetBuffer(bufferSize);
}

void *WorkContext::GetTilingBuffer(uint64_t bufferSize)
{
    if (tilingBufferOffset_ == tilingBuffers_.size()) {
        tilingBufferOffset_ = 0;
    }
    return tilingBuffers_.at(tilingBufferOffset_++)->GetBuffer(bufferSize);
}

void *WorkContext::GetWorkspaceBuffer(uint64_t bufferSize)
{
    if (workspaceBufferOffset_ == workspaceBuffers_.size()) {
        workspaceBufferOffset_ = 0;
    }
    return workspaceBuffers_.at(workspaceBufferOffset_++)->GetBuffer(bufferSize);
}

void *WorkContext::GetIntermediateBuffer(uint64_t bufferSize)
{
    if (intermediateBufferOffset_ == intermediateBuffers_.size()) {
        intermediateBufferOffset_ = 0;
    }
    return intermediateBuffers_.at(intermediateBufferOffset_++)->GetBuffer(bufferSize);
}

uint64_t WorkContext::GetHostTilingBufferRing()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKCONTEXT_HOSTTILING_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return atoi(envStr);
}

uint64_t WorkContext::GetHostTilingBufferSize()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKCONTEXT_HOSTTILING_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}

uint64_t WorkContext::GetTilingBufferRing()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKCONTEXT_TILING_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return atoi(envStr);
}

uint64_t WorkContext::GetTilingBufferSize()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKCONTEXT_TILING_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}

uint64_t WorkContext::GetWorkspaceBufferRing()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKCONTEXT_WORKSPACE_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return atoll(envStr);
}

uint64_t WorkContext::GetWorkspaceBufferSize()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKCONTEXT_WORKSPACE_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}

uint64_t WorkContext::GetIntermediateBufferRing()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKCONTEXT_INTERMEDIATE_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return atoi(envStr);
}

uint64_t WorkContext::GetIntermediateBufferSize()
{
    const char *envStr = std::getenv("ACLTRANSFORMER_WORKCONTEXT_INTERMEDIATE_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}
} // namespace AclTransformer