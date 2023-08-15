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
#ifndef RMSPRENORM_QUANT_RUNNER_BUILDER_H
#define RMSPRENORM_QUANT_RUNNER_BUILDER_H
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/rms_pre_norm_quant.h"
#include "rms_pre_norm_quant_ops_runner.h"

namespace AclTransformer {
class RmsPreNormQuantOpsRunnerBuilder : public RunnerBuilder {
public:
    RmsPreNormQuantOpsRunnerBuilder(const RmsPreNormQuantParam &param) : param_(param) {}
    virtual ~RmsPreNormQuantOpsRunnerBuilder() = default;
    Runner *Build() override { return new RmsPreNormQuantOpsRunner(param_); }

private:
    RmsPreNormQuantParam param_;
};

} // namespace AclTransformer
#endif