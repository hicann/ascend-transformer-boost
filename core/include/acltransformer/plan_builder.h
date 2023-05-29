/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "acltransformer/handle.h"
#include "acltransformer/variant_pack.h"
#include "acltransformer/operation_graph.h"
#include "acltransformer/plan.h"

namespace AclTransformer {
class PlanBuilder {
public:
    Plan *Build(const VariantPack &runInfo, const OperationGraph &opGraph);

private:
    Plan *BuildImpl(const VariantPack &runInfo, const OperationGraph &opGraph);
    void LogOperationGraph(const OperationGraph &opGraph);
    void LogRunnerGraph(const RunnerGraph &runnerGraph);
};
} // namespace AclTransformer
#endif