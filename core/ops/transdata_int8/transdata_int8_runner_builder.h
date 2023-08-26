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
#ifndef TRANSDATAINT8_RUNNER_BUILDER_H
#define TRANSDATAINT8_RUNNER_BUILDER_H
#include "acltransformer/runner_builder.h"
#include "acltransformer/params/transdata_int8.h"
#include "transdata_int8_ops_runner.h"

namespace AclTransformer {
class TransDataInt8OpsRunnerBuilder : public RunnerBuilder {
public:
    TransDataInt8OpsRunnerBuilder(const TransDataInt8Param &param) : param_(param) {}
    virtual ~TransDataInt8OpsRunnerBuilder() = default;
    Runner *Build() override { return new TransDataInt8OpsRunner(param_); }

private:
    TransDataInt8Param param_;
};

} // namespace AclTransformer
#endif