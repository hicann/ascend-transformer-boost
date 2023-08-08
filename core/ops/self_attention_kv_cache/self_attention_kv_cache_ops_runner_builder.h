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
#ifndef SELFATTENTIONKVCACHE_OPS_RUNNER_BUILDER_H
#define SELFATTENTIONKVCACHE_OPS_RUNNER_BUILDER_H
#include <asdops/utils/log/log.h>
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/self_attention_kv_cache.h"
#include <asdops/utils/singleton/singleton.h>
#include "self_attention_kv_cache_ops_chatglm6b_runner.h"
#include "self_attention_kv_cache_ops_llama7b_runner.h"
#include "self_attention_kv_cache_ops_chatglm6b_runner_910a.h"
#include "self_attention_kv_cache_ops_chatglm2_6b_runner.h"
<<<<<<< HEAD
#include "self_attention_kv_cache_ops_llama7b_runner_910a.h"
=======
#include "self_attention_kv_cache_ops_chatglm2_6b_runner_310p.h"
>>>>>>> b361fa1 (feat:add kvcache attention for chatglm2_6b 310p)

namespace AclTransformer {
class SelfAttentionKvCacheOpsRunnerBuilder : public RunnerBuilder {
public:
    SelfAttentionKvCacheOpsRunnerBuilder(const SelfAttentionKvCacheParam &param) : param_(param) {}
    virtual ~SelfAttentionKvCacheOpsRunnerBuilder() = default;
    Runner *Build() override
    {
        if (param_.model == "chatglm6b") {
            if (AsdOps::GetSingleton<Config>().Is910B()) {
                return new SelfAttentionKvCacheOpsChatGlm6bRunner(param_);
            } else {
                return new SelfAttentionKvCacheOpsChatGlm6bRunner910a(param_);
            }
        } else if (param_.model == "llama7b") {
            if (AsdOps::GetSingleton<Config>().Is910B()) {
                return new SelfAttentionKvCacheOpsLlama7bRunner(param_);
            } else {
                return new SelfAttentionKvCacheOpsLlama7bRunner910a(param_);
            }
        } else if (param_.model == "chatglm2_6b") {
            if (AsdOps::GetSingleton<Config>().Is910B()) {
                    return new SelfAttentionKvCacheOpsChatGlm26bRunner(param_); 
                } else {
                    return new SelfAttentionKvCacheOpsChatGlm26bRunner310P(param_);
                }
        } else {
            ASD_LOG(ERROR) << "invalid param_.model:" << param_.model;
            return nullptr;
        }
    }

private:
    SelfAttentionKvCacheParam param_;
};

} // namespace AclTransformer
#endif