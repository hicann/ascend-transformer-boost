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
#ifndef POSITIONEMBEDDING_FUSION_OPS_RUNNER_BUILDER_H
#define POSITIONEMBEDDING_FUSION_OPS_RUNNER_BUILDER_H
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/position_embedding_fusion.h"
#include "position_embedding_fusion_ops_runner.h"
#include "position_embedding_glm2_fusion_ops_runner.h"

namespace AclTransformer {
class PositionEmbeddingFusionOpsRunnerBuilder : public RunnerBuilder {
public:
    explicit PositionEmbeddingFusionOpsRunnerBuilder(const PositionEmbeddingFusionParam &param) : param_(param) {}
    virtual ~PositionEmbeddingFusionOpsRunnerBuilder() = default;
    Runner *Build() override 
    { 
        if (param_.model == "chatglm2_6b" || param_.model == "chatglm2_6b_parallel"){
            return new PositionEmbeddingGlm2FusionOpsRunner(param_);
        }
        return new PositionEmbeddingFusionOpsRunner(param_);
    }

private:
    PositionEmbeddingFusionParam param_;
};
} // namespace AclTransformer
#endif