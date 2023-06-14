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
#ifndef POSITIONEMBEDDING_1D_SPLIT_OPS_RUNNER_BUILDER_H
#define POSITIONEMBEDDING_1D_SPLIT_OPS_RUNNER_BUILDER_H
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/position_embedding_1d_split.h"
#include "position_embedding_1d_split_ops_runner.h"

namespace AclTransformer {
class PositionEmbedding1dSplitOpsRunnerBuilder : public RunnerBuilder {
public:
    PositionEmbedding1dSplitOpsRunnerBuilder(const PositionEmbedding1dSplitParam &param) : param_(param) {}
    virtual ~PositionEmbedding1dSplitOpsRunnerBuilder() = default;
    Runner *Build() override { return new PositionEmbedding1dSplitOpsRunner(param_); }

private:
    PositionEmbedding1dSplitParam param_;
};

} // namespace AclTransformer
#endif