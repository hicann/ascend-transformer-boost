

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
#include "self_attention_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
SelfAttentionOpsRunner::SelfAttentionOpsRunner(const SelfAttentionParam &param)
    : OpsRunner("SelfAttentionOpsRunner", RUNNER_TYPE_SELF_ATTENTION), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionOpsRunner::SelfAttentionOpsRunner called";
}

SelfAttentionOpsRunner::~SelfAttentionOpsRunner() {}
} // namespace AclTransformer
