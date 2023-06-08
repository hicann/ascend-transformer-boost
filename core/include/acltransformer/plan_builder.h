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
#ifndef ACLTRANSFORMER_PLANBUILDER_H
#define ACLTRANSFORMER_PLANBUILDER_H
#include <asdops/utils/status/status.h>
#include "acltransformer/handle.h"
#include "acltransformer/variant_pack.h"
#include "acltransformer/operation_graph.h"
#include "acltransformer/plan.h"

namespace AclTransformer {
class PlanBuilder {
public:
    AsdOps::Status Build(const VariantPack &variantPack, const OperationGraph &opGraph, Plan &plan);

private:
    AsdOps::Status BuildImpl(const VariantPack &variantPack, const OperationGraph &opGraph, Plan &plan);
    void LogOperationGraph(const OperationGraph &opGraph);
    void LogRunnerGraph(const RunnerGraph &runnerGraph);
};
} // namespace AclTransformer
#endif