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
#ifndef LINERA_OPS_RUNNER_BUILDER_H
#define LINERA_OPS_RUNNER_BUILDER_H
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "acltransformer/runner/runner_builder.h"
#include "acltransformer/params/linear.h"
#include "linear_ops_runner_910a.h"
#include "linear_ops_runner_910b.h"

namespace AclTransformer {
class LinearOpsRunnerBuilder : public RunnerBuilder {
public:
    LinearOpsRunnerBuilder(const LinearParam &param) : param_(param)
    {
        ASD_LOG(INFO) << "LinearOperation::LinearOperation called";
        }
    virtual ~LinearOpsRunnerBuilder() = default;
    Runner *Build() override
    {
        if (AsdOps::GetSingleton<Config>().Is910B()) {
            return new LinearOpsRunner910B(param_);
        } else {
            return new LinearOpsRunner910A(param_);
        }
    }

private:
    LinearParam param_;
};

} // namespace AclTransformer
#endif