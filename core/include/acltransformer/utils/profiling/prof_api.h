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
#ifndef PROFILING_API_H
#define PROFILING_API_H
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

using VOID_PTR = void *;

const uint32_t MSPROF_REPORT_DATA_MAGIC_NUM = 0x5a5a;

const uint16_t MSPROF_REPORT_ACL_LEVEL = 20000;
const uint16_t MSPROF_REPORT_NODE_LEVEL = 10000;

const uint32_t MSPROF_REPORT_ACL_OTHERS_BASE_TYPE = 0x40000;

const uint32_t MSPROF_REPORT_NODE_BASIC_INFO_TYPE = 0;
const uint32_t MSPROF_REPORT_NODE_LAUNCH_TYPE = 5;

enum MsprofGeTaskType {
    MSPROF_GE_TASK_TYPE_AI_CORE = 0,
    MSPROF_GE_TASK_TYPE_AI_CPU,
    MSPROF_GE_TASK_TYPE_AIV,
    MSPROF_GE_TASK_TYPE_WRITE_BACK,
    MSPROF_GE_TASK_TYPE_MIX_AIC,
    MSPROF_GE_TASK_TYPE_MIX_AIV,
    MSPROF_GE_TASK_TYPE_FFTS_PLUS,
    MSPROF_GE_TASK_TYPE_DSA,
    MSPROF_GE_TASK_TYPE_DVPP,
    MSPROF_GE_TASK_TYPE_HCCL,
    MSPROF_GE_TASK_TYPE_INVALID
};

struct MsProfApi {
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t reserve;
    uint64_t beginTime;
    uint64_t endTime;
    uint64_t itemId;
};

struct MsprofRuntimeTrack {
    uint16_t deviceId;
    uint16_t streamId;
    uint32_t taskId;
    uint64_t taskType;
};

#pragma pack(1)
struct MsprofNodeBasicInfo {
    uint64_t opName;
    uint32_t taskType;
    uint64_t opType;
    uint32_t blockDim;
    uint32_t opFlag;
};
#pragma pack()

const uint16_t MSPROF_COMPACT_INFO_DATA_LENGTH = 40;
struct MsprofCompactInfo {
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    union {
        uint8_t info[MSPROF_COMPACT_INFO_DATA_LENGTH];
        MsprofRuntimeTrack runtimeTrack;
        MsprofNodeBasicInfo nodeBasicInfo;
    } data;
};

MSVP_PROF_API int32_t MsprofReportApi(uint32_t agingFlag, const MsProfApi *api);

MSVP_PROF_API int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

MSVP_PROF_API uint64_t MsprofGetHashId(const char *hashInfo, size_t length);

MSVP_PROF_API uint64_t MsprofSysCycleTime();

#ifdef __cplusplus
}
#endif
#endif