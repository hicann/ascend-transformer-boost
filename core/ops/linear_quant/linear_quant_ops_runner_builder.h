
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
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/linear_quant.h"
#include "linear_quant_ops_runner.h"

namespace AclTransformer {
class LinearQuantOpsRunnerBuilder : public RunnerBuilder {
public:
    explicit LinearQuantOpsRunnerBuilder(const LinearQuantParam &param) : param_(param)
    {
        ASD_LOG(INFO) << "LinearOQuantperation::LinearQuantOperation called";
    }
    virtual ~LinearQuantOpsRunnerBuilder() = default;
    Runner *Build() override { return new LinearQuantOpsRunner(param_); }

private:
    LinearQuantParam param_;
};
} // namespace AclTransformer
#endif