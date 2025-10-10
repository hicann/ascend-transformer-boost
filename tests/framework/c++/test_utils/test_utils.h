/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ATB_UNIT_TEST_UTIL_H
#define ATB_UNIT_TEST_UTIL_H
#include "atb/context.h"

namespace atb {
struct FilteredStats {
    double mean;
    double stddev;
    size_t count;
    size_t validCount;
    double filteredMean;
};

bool IsNot910B(void *obj);
bool IsNot310P(void *obj);
Context* CreateContextAndExecuteStream();

/**
 * @brief 计算数据集的统计信息，包括原始均值、标准差，以及过滤掉3σ范围外异常值后的均值。
 *
 * 该函数首先计算输入数据的原始均值(mean)和标准差(stddev)。
 * 然后，它使用 3σ 准则（即[mean - 3 * stddev, mean + 3 * stddev] 区间）来过滤异常值。
 * 最后，它计算并返回有效数据点的数量(validCount)和过滤后的均值(filtered_mean)。
 *
 * @param data 包含原始数据（例如：时间测量值等）的vector。
 * 数据类型为 uint64_t。
 * @return FilteredStats 结构体，包含计算得到的各项统计数据。
 * @retval FilteredStats::mean 原始数据集的算术平均值。
 * @retval FilteredStats::stddev 原始数据集的标准差。
 * @retval FilteredStats::validCount 过滤后，落在3σ范围内的有效数据点数量。
 * @retval FilteredStats::filteredMean 过滤后的有效数据集的算术平均值。如果validCount为0，则为0.0。
 */
Stats CalculateFilteredMean(const std::vector<uint64_t> &data);

std::string FilteredStatsToString(const FilteredStats &stat);
}
#endif