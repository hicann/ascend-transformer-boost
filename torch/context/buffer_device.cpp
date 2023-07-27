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
#include "buffer_device.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/statistic.h"

namespace AclTransformer {
BufferDevice::BufferDevice(uint64_t bufferSize) : bufferSize_(bufferSize)
{
    ASD_LOG(INFO) << "BufferDevice::BufferDevice called, bufferSize:" << bufferSize;
    if (bufferSize_ > 0) {
        ASD_LOG(FATAL) << "BufferDevice::GetBuffer AsdRtMemMallocDevice bufferSize:" << bufferSize_;
        int st = AsdRtMemMallocDevice((void **)&buffer_, bufferSize_, ASDRT_MEM_DEFAULT);
        if (st != ASDRT_SUCCESS) {
            ASD_LOG(FATAL) << "BufferDevice::GetBuffer AsdRtMemMallocDevice fail, ret:" << st;
        }
    }
}

BufferDevice::~BufferDevice() { Free(); }

void *BufferDevice::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= bufferSize_) {
        ASD_LOG(INFO) << "BufferDevice::GetBuffer bufferSize:" << bufferSize << " <= bufferSize_:" << bufferSize_
                      << ", not new device mem";
        return buffer_;
    }

    Free();

    ASD_LOG(FATAL) << "BufferDevice::GetBuffer AsdRtMemMallocDevice bufferSize:" << bufferSize;
    int st = AsdRtMemMallocDevice((void **)&buffer_, bufferSize, ASDRT_MEM_DEFAULT);
    if (st != ASDRT_SUCCESS) {
        ASD_LOG(ERROR) << "BufferDevice::GetBuffer AsdRtMemMallocDevice fail, ret:" << st;
        return nullptr;
    }
    ASD_LOG(INFO) << "BufferDevice::GetBuffer AsdRtMemMallocDevice success, buffer:" << buffer_;
    bufferSize_ = bufferSize;
    return buffer_;
}

void BufferDevice::Free()
{
    if (buffer_) {
        ASD_LOG(INFO) << "BufferDevice::GetBuffer AsdRtMemFreeDevice buffer:" << buffer_
                      << ", bufferSize:" << bufferSize_;
        AsdRtMemFreeDevice(buffer_);
        buffer_ = nullptr;
        bufferSize_ = 0;
    }
}
} // namespace AclTransformer