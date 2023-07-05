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
#ifndef SELFATTENTIONKVCACHEFUSION_OPS_RUNNER_BUILDER_H
#define SELFATTENTIONKVCACHEFUSION_OPS_RUNNER_BUILDER_H
#include <asdops/utils/log/log.h>
#include "acltransformer/runner/runner_builder.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "self_attention_kv_cache_fusion_ops_chatglm6b_runner.h"

namespace AclTransformer {
class SelfAttentionKvCacheFusionOpsRunnerBuilder : public RunnerBuilder {
public:
    explicit SelfAttentionKvCacheFusionOpsRunnerBuilder(const SelfAttentionKvCacheFusionParam &param) : param_(param)
    {}
    virtual ~SelfAttentionKvCacheFusionOpsRunnerBuilder() = default;
    Runner *Build() override
    {
        return new SelfAttentionKvCacheFusionOpsChatGlm6bRunner(param_);
    }

private:
    SelfAttentionKvCacheFusionParam param_;
};
} // namespace AclTransformer
#endif