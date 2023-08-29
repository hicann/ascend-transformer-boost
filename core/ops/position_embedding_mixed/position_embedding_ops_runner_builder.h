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
#ifndef POSITIONEMBEDDING_OPS_RUNNER_BUILDER_H
#define POSITIONEMBEDDING_OPS_RUNNER_BUILDER_H
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/position_embedding.h"
#include "position_embedding_ops_runner.h"
#include "position_embedding_1d_ops_llama7b_runner.h"
#include "position_embedding_1d_ops_runner.h"
#include "position_embedding_1d_fusion_ops_runner.h"
#include "position_embedding_ops_glm2_runner.h"
#include "position_embedding_ops_gptneox20b_runner.h"

namespace AclTransformer {
class PositionEmbeddingOpsRunnerBuilder : public RunnerBuilder {
public:
    PositionEmbeddingOpsRunnerBuilder(const PositionEmbeddingParam &param) : param_(param) {}
    virtual ~PositionEmbeddingOpsRunnerBuilder() = default;
    Runner *Build() override
    {
        if (param_.model == "chatglm2_6b") {
            return new PositionEmbeddingOpsGlm2Runner(param_);
        } else if (param_.model == "gptneox20b") {
            return new PositionEmbeddingOpsGptNeox20bRunner(param_);
        } else if (param_.model == "llama7b") {
            return new PositionEmbedding1dOpsLlama7bRunner(param_);
        } else if (param_.is2d) {
            return new PositionEmbeddingOpsRunner(param_);
        } else {
            if (param_.isFusion) {
                return new PositionEmbedding1dMixedFusionOpsRunner(param_);
            } else {
                return new PositionEmbedding1dOpsRunner(param_);
            }
        }
    }

private:
    PositionEmbeddingParam param_;
};

} // namespace AclTransformer
#endif