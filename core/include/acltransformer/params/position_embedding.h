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
#ifndef ACLTRANSFOERM_PARAMS_POSITION_EMBEDDING_H
#define ACLTRANSFOERM_PARAMS_POSITION_EMBEDDING_H
namespace AclTransformer {
struct PositionEmbeddingParam {
    bool is2d = true;
    int64_t headNum = 0;
    int64_t numHeadPerPartition = 0;
    int64_t hiddenSizePerHead = 0;
    int64_t numGroupsPerPartition = 0;
    int64_t dk = 0;
    float rotaryPct = 0.25;
    std::string model = "chatglm";
};
} // namespace AclTransformer
#endif