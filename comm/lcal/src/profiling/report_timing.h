/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REPORT_TIMING_H
#define REPORT_TIMING_H
#include <cstring>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <security.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <toolchain/slog.h>
#include <toolchain/prof_api.h>
#include <toolchain/prof_common.h>
#include <hccl_types.h>
#include <mki/utils/log/log.h>

namespace Lcal {
class ReportTiming {
public:
    static constexpr uint64_t PROF_TASK_TIME_DUMP = 0x000100000000ULL;
    ReportTiming() = delete;
    explicit ReportTiming(const char *opName, int commDomain, int64_t count =0,
                          HcclDataType dataType = HCCL_DATA_TYPE_RESERVED)
        : opName_(opName), typeMix_(false), count_(count), dataType_(dataType)
    {
        InitProfiling(commDomain);
    }

    explicit ReportTiming(const char *opName, uint32_t blockDim)
        : opName_(opName), blockDim_(blockDim), typeMix_(true)
    {
        InitProfiling(0);
    }

    explicit ReportTiming(const char *opName, const int32_t rankId, const bool isReporting, uint8_t *dumpAddr,
        const aclrtStream stream) : opName_(opName), rankId_(rankId), isReporting_(isReporting),
        dumpAddr_(dumpAddr), stream_(stream)
    {
        moduleId_ = DUMP_MODULE_ID;
        InitProfiling(0);
    }

    ~ReportTiming()
    {
        MKI_LOG(DEBUG) << "ReportTiming " << __LINE__ << " ~ReportTiming() " <<
            " isReporting_:" << isReporting_ << " profEnable_:" << profEnable_;
        if (profEnable_ && isReporting_) {
            ReportMsprofData();
        }

        if (!isReporting_) {
            ProfilingStatus(RESET_STATUS);
        }
    }

    void InitProfiling(int commDomain)
    {
        if (ProfilingStatus() == -1) {
            ProfilingStatus(0);
            MKI_LOG(INFO) << "MsprofRegisterCallback start!";
            if (MsprofRegisterCallback(moduleId_, ProfHandle) != 0) {
                MKI_LOG(ERROR) << "MsprofRegisterCallback fail!";
            }
        }

        MKI_LOG(DEBUG) << "InitProfiling " << __LINE__ << "ProfilingStatus()" << ProfilingStatus() <<
            " isReporting_:" << isReporting_;
        if (ProfilingStatus() > 0) {
            ParamsInit(commDomain);
        }
        MKI_LOG(DEBUG) << "InitProfiling " << __LINE__ << "ProfilingStatus()" << ProfilingStatus() <<
            " isReporting_:" << isReporting_ << " profEnable_:" << profEnable_;
    }




private:
    static constexpr uint64_t PROF_TASK_TIME_L0 = 0x00000800ULL;
    static constexpr uint64_t PROF_TASK_TIME_L1 = 0x00000002ULL;
    static constexpr int32_t DUMP_MODULE_ID = 61;
    static constexpr int32_t RESET_STATUS = -2;
    uint64_t beginTime_ = 0;
    uint16_t endTime_ = 0;
    const char *opName_ = nullptr;
    uint32_t blockDim_ = 0;
    uint64_t nameHash_ = 0;
    uint64_t groupHash_ = 0;
    uint64_t naHash_ = 0;
    bool typeMix_ = false;
    long tid_ = 0;
    bool profEnable_ = false;
    int64_t count_ = 0;
    uint8_t dataType = HCCL_DATA_TYPE_RESERVED;
    int32_t rankId_ = 0;
    bool isReporting_ = true;
    uint8_t *dumpAddr_ = nullptr;
    aclrtStream stream_ = nullptr;
    int32_t moduleId_ = INVALID_MODULE_ID;
};



}

















#endif