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
#ifndef SELF_ATTETION_KV_CACHE_TORCH_RUNNER_BUILDER_H
#define SELF_ATTETION_KV_CACHE_TORCH_RUNNER_BUILDER_H
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/self_attention_kv_cache.h"
#include "self_attention_kv_cache_torch_runner.h"

namespace AclTransformer {
class SelfAttentionKvCacheTorchRunnerBuilder : public RunnerBuilder {
public:
    SelfAttentionKvCacheTorchRunnerBuilder(const SelfAttentionKvCacheParam &param) : param_(param) {}
    virtual ~SelfAttentionKvCacheTorchRunnerBuilder() = default;
    Runner *Build() override { return new SelfAttentionKvCacheTorchRunner(param_); }

private:
    SelfAttentionKvCacheParam param_;
};

} // namespace AclTransformer
#endif