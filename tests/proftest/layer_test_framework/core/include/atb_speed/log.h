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

#ifndef ATB_SPEED_LOG_H
#define ATB_SPEED_LOG_H

#include <iostream>
#include <sstream>
#include <mutex>

#include <sys/stat.h>

#include "nlohmann/json.hpp"

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (auto& el : vec) {
        os << el << ',';
    }
    return os;
}

#define ATB_SPEED_LOG(msg, ...)                                                                                        \
    do {                                                                                                               \
        std::ostringstream oss;                                                                                        \
        oss << msg;                                                                                                    \
    } while (0)

#define ATB_SPEED_LOG_DEBUG(msg, ...) ATB_SPEED_LOG(msg, __VA_ARGS__)

#define ATB_SPEED_LOG_INFO(msg, ...) ATB_SPEED_LOG(msg, __VA_ARGS__)

#define ATB_SPEED_LOG_WARN(msg, ...) ATB_SPEED_LOG(msg, __VA_ARGS__)

#define ATB_SPEED_LOG_ERROR(msg, ...) ATB_SPEED_LOG(msg, __VA_ARGS__)

#define ATB_SPEED_LOG_FATAL(msg, ...) ATB_SPEED_LOG(msg, __VA_ARGS__)

#endif