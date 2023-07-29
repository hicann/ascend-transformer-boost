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
#ifndef FFN_OPS_RUNNER_BUILDER_H
#define FFN_OPS_RUNNER_BUILDER_H
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/ffn.h"
#include "ffn_ops_runner_910a.h"
#include "ffn_ops_runner.h"

namespace AclTransformer {
class FfnOpsRunnerBuilder : public RunnerBuilder {
public:
    FfnOpsRunnerBuilder(const FfnParam &param) : param_(param) {}
    virtual ~FfnOpsRunnerBuilder() = default;
    Runner *Build() override 
    { 
        if (AsdOps::GetSingleton<Config>().Is910B()) {
            return new FfnOpsRunner(param_);
        } else {
            return new FfnOpsRunner910A(param_);
        }
    }

private:
    FfnParam param_;
};

} // namespace AclTransformer
#endif