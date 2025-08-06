/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef ATB_SPEED_MODELS_ALL_TO_ALL_DISPATCH_OPERATION_H
#define ATB_SPEED_MODELS_ALL_TO_ALL_DISPATCH_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/comm.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
struct AllToAllDispatchParam {
    int topk = 1;
    int numOfExperts = 8;
    std::string backend = "hccl";
    HcclComm hcclComm = nullptr;
    bool hasMoeEp = false;
    int moeEpRank = 0;
    int moeEpSize = 1;
    std::string moeEpDomain = "";
    std::string moeEpRankTableFile = "";

    bool hasMlpTp = false;
    int mlpTpRank = 0;
    int mlpTpSize = 1;
    std::string mlpTpDomain = "";
    std::string mlpTpRankTableFile = "";
};

atb::Status CreateAllToAllDispatchOperation(const AllToAllDispatchParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif