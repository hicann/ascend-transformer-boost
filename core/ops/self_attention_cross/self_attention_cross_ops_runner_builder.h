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
#ifndef SELFATTENTIONCROSS_OPS_RUNNER_BUILDER_H
#define SELFATTENTIONCROSS_OPS_RUNNER_BUILDER_H
#include <asdops/utils/log/log.h>
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/self_attention_cross.h"
#include <asdops/utils/singleton/singleton.h>
#include "self_attention_cross_ops_llama7badapter_runner_310p.h"
#include "self_attention_cross_ops_llama7badapter_runner_encoder_310p.h"


namespace AclTransformer {
class SelfAttentionCrossOpsRunnerBuilder : public RunnerBuilder {
public:
    SelfAttentionCrossOpsRunnerBuilder(const SelfAttentionCrossParam &param) : param_(param) {}
    virtual ~SelfAttentionCrossOpsRunnerBuilder() = default;
    Runner *Build() override
    {
        if (param_.model == "llama_adapter") {
            return new SelfAttentionCrossOpsLlama7bAdapterRunner310p(param_);
        } else if (param_.model == "llama_adapter_encoder") {
            return new SelfAttentionCrossOpsLlama7bAdapterRunnerEncoder310p(param_);
        }else {
            ASD_LOG(ERROR) << "invalid param_.model:" << param_.model;
            return nullptr;
        }
    }

private:
    SelfAttentionCrossParam param_;
};

} // namespace AclTransformer
#endif