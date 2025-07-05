/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ATB_OPS_PAGED_ATTENTION_910_95_TILING_H
#define ATB_OPS_PAGED_ATTENTION_910_95_TILING_H

#include <mki/bin_handle.h>
#include <mki/kernel_info.h>
#include <mki/launch_param.h>
#include <mki/utils/status/status.h>

using namespace Mki;

namespace AtbOps {
Status PagedAttentionBaseAscend91095Tiling(const std::string &kernelName, const LaunchParam &launchParam,
                                           KernelInfo &kernelInfo, const Mki::BinHandle &binHandle);
Status PagedAttentionW8A16Ascend91095Tiling(const std::string &kernelName, const LaunchParam &launchParam,
                                            KernelInfo &kernelInfo, const Mki::BinHandle &binHandle);
} // namespace AtbOps
#endif // ATB_OPS_PAGED_ATTENTION_910_95_TILING_H
