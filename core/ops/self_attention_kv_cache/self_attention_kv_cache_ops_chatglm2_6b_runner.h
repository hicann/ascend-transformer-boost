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
#ifndef SELFATTENTIONKVCACHE_OPS_CHATGLM2_6B_RUNNER_H
#define SELFATTENTIONKVCACHE_OPS_CHATGLM2_6B_RUNNER_H
#include "acltransformer/base/ops_runner.h"
#include "acltransformer/params/self_attention_kv_cache.h"

namespace AclTransformer {
class SelfAttentionKvCacheOpsChatGlm26bRunner : public OpsRunner {
public:
    SelfAttentionKvCacheOpsChatGlm26bRunner(const SelfAttentionKvCacheParam &param);
    virtual ~SelfAttentionKvCacheOpsChatGlm26bRunner();

private:
    SelfAttentionKvCacheParam param_;
};

} // namespace AclTransformer
#endif