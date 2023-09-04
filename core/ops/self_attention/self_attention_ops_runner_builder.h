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
#ifndef SELF_ATTETION_OPS_RUNNER_BUILDER_H
#define SELF_ATTETION_OPS_RUNNER_BUILDER_H
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/self_attention.h"
#include "self_attention_ops_openbert_runner.h"
#include "self_attention_ops_chatglm6b_runner.h"
#include "self_attention_ops_chatglm2_6b_runner.h"
#include "self_attention_ops_chatglm2_6b_runner_310p.h"
#include <asdops/utils/log/log.h>
#include "self_attention_ops_chatglm6b_runner_910a.h"
#include "self_attention_ops_llama7b_runner.h"
#include "self_attention_ops_baichuan1_7b_runner_910a.h"
#include "self_attention_ops_baichuan2_13b_runner_910a.h"
#include "self_attention_ops_gptneox20b_runner.h"

namespace AclTransformer {
class SelfAttentionOpsRunnerBuilder : public RunnerBuilder {
public:
    SelfAttentionOpsRunnerBuilder(const SelfAttentionParam &param) : param_(param) {}
    virtual ~SelfAttentionOpsRunnerBuilder() = default;
    Runner *Build() override { 
        if (param_.model == "openbert") {
            return new SelfAttentionOpsOpenbertRunner(param_);
        } else if (param_.model == "chatglm6b" || param_.model == "glm130b") {
            if (AsdOps::GetSingleton<Config>().Is910B()) {
                return new SelfAttentionOpsChatglm6bRunner(param_); 
            } else {
                return new SelfAttentionOpsChatglm6bRunner910a(param_);
            }
        } else if (param_.model == "chatglm2_6b" || param_.model == "chatglm2_6b_parallel") {
            if (AsdOps::GetSingleton<Config>().Is910B()) {
                return new SelfAttentionOpsChatglm26bRunner(param_);
            } else {
                return new SelfAttentionOpsChatglm26bRunner310P(param_);
            }
        } else if (param_.model == "llama7b" || param_.model == "llama13b" || param_.model == "llama65b") {
            if (AsdOps::GetSingleton<Config>().Is910B()) {
                return new SelfAttentionOpsLlama7bRunner(param_); 
            } else {
                return nullptr;
            }
        } else if (param_.model == "baichuan1_7b") {
            if (AsdOps::GetSingleton<Config>().Is910B()) {
                ASD_LOG(ERROR) << "invalid param_.model:" << param_.model << " for 910b";
                return nullptr;
            } else {
                return new SelfAttentionOpsBaiChuan17BRunner910a(param_);
            }
        } else if (param_.model == "baichuan2_13b") {
            if (AsdOps::GetSingleton<Config>().Is910B()) {
                ASD_LOG(ERROR) << "invalid param_.model:" << param_.model << " for 910b";
                return nullptr;
            } else {
                return new SelfAttentionOpsBaiChuan213BRunner910a(param_);
            }
        } else if (param_.model == "gptneox20b") {
            return new SelfAttentionOpsGptNeox20bRunner(param_);
        } else {
            ASD_LOG(ERROR) << "invalid param_.model:" << param_.model;
            return nullptr;
        }
    }

private:
    SelfAttentionParam param_;
};

} // namespace AclTransformer
#endif