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
#ifndef ACLTRANSFORMER_OPERATIONGRAPH_H
#define ACLTRANSFORMER_OPERATIONGRAPH_H
#include <vector>
#include <cstdint>
#include <functional>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/operation.h"

namespace AclTransformer {
using OperationGraphNodeViewFunc =
    std::function<void(const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)>;

struct OperationGraphNode {
    const Operation *operation = nullptr;
    AsdOps::SVector<uint64_t> inTensorIds;
    AsdOps::SVector<uint64_t> outTensorIds;
    AsdOps::SVector<OperationGraphNodeViewFunc> inTensorViewFuncs;
};

struct OperationGraph {
    uint64_t inTensorSize = 0;
    uint64_t outTensorSize = 0;
    uint64_t intermediateTensorSize = 0;
    AsdOps::SVector<OperationGraphNode, 64> nodes;
    std::string ToString() const;
};
} // namespace AclTransformer
#endif
