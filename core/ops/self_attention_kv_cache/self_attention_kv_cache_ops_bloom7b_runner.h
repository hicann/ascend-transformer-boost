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
#ifndef SELFATTENTIONKVCACHE_OPS_BLOOM7B_RUNNER_H
#define SELFATTENTIONKVCACHE_OPS_BLOOM7B_RUNNER_H
#include "acltransformer/base/ops_runner.h"
#include "acltransformer/params/self_attention_kv_cache.h"

namespace AclTransformer {
class SelfAttentionKvCacheOpsBloom7bRunner : public OpsRunner {
public:
    SelfAttentionKvCacheOpsBloom7bRunner(const SelfAttentionKvCacheParam &param);
    virtual ~SelfAttentionKvCacheOpsBloom7bRunner();

private:
    SelfAttentionKvCacheParam param_;
    AsdOps::SVector<int64_t> oriDimA_;
    AsdOps::SVector<int64_t> oriDimB_;
    AsdOps::SVector<int64_t> oriDimC_;
    AsdOps::SVector<int64_t> oriDimD_;
    AsdOps::SVector<int64_t> oriDimE_;
    AsdOps::SVector<int64_t> oriDimF_;
    AsdOps::SVector<int64_t> oriDimG_;
    AsdOps::SVector<int64_t> oriDimH_;   
    AsdOps::SVector<int64_t> oriDimI_;
    AsdOps::SVector<int64_t> oriDimJ_;      
    std::size_t oriSize_ = 3;
};

} // namespace AclTransformer
#endif