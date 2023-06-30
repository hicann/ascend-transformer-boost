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
#ifndef ACLTRANSFORMER_RUNNERTYPE_H
#define ACLTRANSFORMER_RUNNERTYPE_H

namespace AclTransformer {
enum RunnerType {
    RUNNER_TYPE_UNDEFINED = -1,
    RUNNER_TYPE_ADD = 0,
    RUNNER_TYPE_ADD_NORM,
    RUNNER_TYPE_FFN,
    RUNNER_TYPE_LINEAR,
    RUNNER_TYPE_MLP,
    RUNNER_TYPE_NORM,
    RUNNER_TYPE_POSITION_EMBEDDING_1D_SPLIT,
    RUNNER_TYPE_POSITION_EMBEDDING_2D_MIXED,
    RUNNER_TYPE_RMS_NORM,
    RUNNER_TYPE_SELF_ATTENTION,
    RUNNER_TYPE_SELF_ATTENTION_KV_CACHE,
    RUNNER_TYPE_SELF_ATTENTION_KV_FUSION_CACHE,
    RUNNER_TYPE_TRANSPOSE,
    RUNNER_TYPE_MAX
};
} // namespace AclTransformer
#endif