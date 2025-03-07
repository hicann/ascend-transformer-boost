/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/utils_internal.h"
#include <unistd.h>
#include <syscall.h>
#include <cmath>
#include <limits>
#include "atb/utils/log.h"

namespace atb {
bool UtilsInternal::IsFloatEqual(float lh, float rh)
{
    return (std::fabs(lh - rh) <= std::numeric_limits<float>::epsilon());
}

bool UtilsInternal::IsDoubleEqual(double lh, double rh)
{
    return (std::fabs(lh - rh) <= std::numeric_limits<double>::epsilon());
}

int32_t UtilsInternal::GetCurrentThreadId()
{
    int32_t tid = static_cast<int32_t>(syscall(SYS_gettid));
    if (tid == -1) {
        ATB_LOG(ERROR) << "get tid failed, errno: " << errno;
    }
    return tid;
}

int32_t UtilsInternal::GetCurrentProcessId()
{
    int32_t pid = static_cast<int32_t>(syscall(SYS_getpid));
    if (pid == -1) {
        ATB_LOG(ERROR) << "get pid failed, errno: " << errno;
    }
    return pid;
}

int64_t UtilsInternal::AlignUp(int64_t dim, int64_t align)
{
    if (align == 0) {
        return -1;
    }
    return (dim + align - 1) / align * align;
}
} // namespace atb