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
#ifndef ACLTRANSFORMER_RUNNER_RUNNER_GRAPH_H
#define ACLTRANSFORMER_RUNNER_RUNNER_GRAPH_H
#include <map>
#include <set>
#include <functional>
#include <memory>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/runner/runner.h"
#include "acltransformer/operation.h"

namespace AclTransformer {
using RunnerGraphNodeViewFunc =
    std::function<void(const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)>;

struct RunnerGraphNode {
    const Operation *operation = nullptr;
    std::shared_ptr<Runner> runner;
    AsdOps::SVector<AsdOps::Tensor *> inTensors;
    AsdOps::SVector<AsdOps::Tensor *> outTensors;
    AsdOps::SVector<RunnerGraphNodeViewFunc> inTensorViewFuncs;
    VariantPack variantPack;
};

struct RunnerGraph {
    AsdOps::SVector<AsdOps::Tensor> inTensors;
    AsdOps::SVector<AsdOps::Tensor> outTensors;
    AsdOps::SVector<AsdOps::Tensor> internalTensors;
    AsdOps::SVector<RunnerGraphNode> nodes;
    std::map<AsdOps::Tensor *, uint64_t> tensorMaxNodeIdMap;
    std::map<uint64_t, std::set<AsdOps::Tensor *>> maxNodeIdTensorMap;
    std::string ToString() const;
    void InitTensorMaxNodeMap();
};
} // namespace AclTransformer
#endif