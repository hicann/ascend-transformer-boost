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
#ifndef MLP_OPS_RUNNER_BUILDER_H
#define MLP_OPS_RUNNER_BUILDER_H
#include <asdops/utils/log/log.h>
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/mlp.h"
#include "mlp_ops_runner.h"
#include "mlp_ops_glm130b_runner.h"
#include "mlp_ops_glm2_6b_runner.h"

namespace AclTransformer {
class MlpOpsRunnerBuilder : public RunnerBuilder {
public:
    MlpOpsRunnerBuilder(const MlpParam &param) : param_(param) {}
    virtual ~MlpOpsRunnerBuilder() = default;
    Runner *Build() override
    {
        if (param_.model == "glm130b") {
            return new MlpOpsGlm130bRunner(param_);
        } else if (param_.model == "chatglm2_6b"){
            return new MlpOpsGlm2Runner(param_);
        }else {
            return new MlpOpsRunner(param_);
        }
    }

private:
    MlpParam param_;
};

} // namespace AclTransformer
#endif