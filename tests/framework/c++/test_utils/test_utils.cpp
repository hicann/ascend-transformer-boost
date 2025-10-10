/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "test_utils.h"
#include <string>
#include <acl/acl_rt.h>
#include <atb/utils/log.h>
#include "atb/context/context_base.h"
#include "atb/types.h"

namespace atb {
bool IsNot910B(void *obj)
{
    const char *socName = aclrtGetSocName();
    if (!socName) {
        ATB_LOG(ERROR) << "aclrtGetSocName failed!";
        return false;
    }
    ATB_LOG(INFO) << "SocVersion:" << std::string(socName);
    return std::string(socName).find("Ascend910B") == std::string::npos &&
        std::string(socName).find("Ascend910_93") == std::string::npos;
}

bool IsNot310P(void *obj)
{
    const char *socName = aclrtGetSocName();
    if (!socName) {
        ATB_LOG(ERROR) << "aclrtGetSocName failed!";
        return false;
    }
    ATB_LOG(INFO) << "SocVersion:" << std::string(socName);
    return std::string(socName).find("Ascend310P") == std::string::npos;
}

Context *CreateContextAndExecuteStream()
{
    int deviceId = -1;
    ATB_LOG(INFO) << "aclrtGetDevice start";
    int ret = aclrtGetDevice(&deviceId);
    ATB_LOG(INFO) << "aclrtGetDevice ret:" << ret << ", deviceId:" << deviceId;
    if (ret != 0 || deviceId < 0) {
        const char *envStr = std::getenv("SET_NPU_DEVICE");
        deviceId = (envStr != nullptr) ? atoi(envStr) : 0;
        ATB_LOG(INFO) << "aclrtSetDevice deviceId:" << deviceId;
        ret = aclrtSetDevice(deviceId);
        ATB_LOG_IF(ret != 0, ERROR) << "aclrtSetDevice fail, ret:" << ret << ", deviceId:" << deviceId;
    }

    atb::Context *context = nullptr;
    Status st = atb::CreateContext(&context);
    ATB_LOG_IF(st != 0, ERROR) << "CreateContext fail";
    aclrtStream stream = nullptr;
    st = aclrtCreateStream(&stream);
    ATB_LOG_IF(st != 0, ERROR) << "aclrtCreateStream fail";
    context->SetExecuteStream(stream);
    return context;
}

FilteredStats CalculateFilteredMean(const std::vector<uint64_t> &data)
{
    if (data.empty()) {
        ATB_LOG(ERROR) << "data to be filtered is empty. Returning 0";
        return {0, 0, 0, 0, 0};
    }
    FilteredStats stats;
    stats.count = data.size();
    // 计算原始均值
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    stats.mean = sum / stats.count;

    // 计算标准差
    double l2NormSum = 0.0;
    for (double val : data) {
        l2NormSum += (val - stats.mean) * (val - stats.mean);
    }
    stats.stddev = std::sqrt(l2NormSum / stats.count);

    // 定义有效范围 (3σ 区间)
    double lower = stats.mean - 3 * stats.stddev;
    double upper = stats.mean + 3 * stats.stddev;

    // 过滤数据
    std::vector<double> filteredData;
    std::copy_if(data.begin(), data.end(), std::back_inserter(filteredData),
                 [lower, upper](double x) { return x >= lower && x <= upper; });

    stats.validCount = filteredData.size();

    // 计算过滤后的均值
    if (stats.validCount == 0) {
        stats.filtered_mean = 0.0; // 或抛出异常
    } else {
        stats.filtered_mean = std::accumulate(filteredData.begin(), filteredData.end(), 0.0) / stats.validCount;
    }
    return stats;
}

std::string FilteredStatsToString(const FilteredStats &stat)
{
    std::stringstream ss;
    ss << "origin count: " << stat.count << "\tvalid count: " << stat.validCount << "\torigin average value: "
        << stat.mean << "\tstandard deviation" << stat.stddev << "\tfiltered average (±3σ)" << stat.filteredMean;
    return ss.str();
}
} // namespace atb
