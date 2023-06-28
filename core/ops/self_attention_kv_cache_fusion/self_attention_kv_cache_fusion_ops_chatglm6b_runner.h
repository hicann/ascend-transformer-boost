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
#ifndef SELFATTENTIONKVCACHEFUSION_OPS_CHATGLM6B_RUNNER_H
#define SELFATTENTIONKVCACHEFUSION_OPS_CHATGLM6B_RUNNER_H
#include "acltransformer/base/ops_runner.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"

namespace AclTransformer {
class SelfAttentionKvCacheFusionOpsChatGlm6bRunner : public OpsRunner {
public:
    explicit SelfAttentionKvCacheFusionOpsChatGlm6bRunner(const SelfAttentionKvCacheFusionParam &param);
    virtual ~SelfAttentionKvCacheFusionOpsChatGlm6bRunner();
private:
    SelfAttentionKvCacheFusionParam param_;
    const std::size_t inTensSize = 9;
    const std::size_t nodeSize = 3;
    const std::size_t index2 = 2;
    const std::size_t index3 = 3;
    const std::size_t index4 = 4;
    const std::size_t index5 = 5;
    const std::size_t index6 = 6;
    const std::size_t index7 = 7;
    const std::size_t index8 = 8;
};
} // namespace AclTransformer
#endif