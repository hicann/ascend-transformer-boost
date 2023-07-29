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
#ifndef ACLTRANSFORMER_GRAPH_OPERATION_H
#define ACLTRANSFORMER_GRAPH_OPERATION_H
#include <string>
#include <functional>
#include <memory>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/operation.h"

namespace AclTransformer {
class GraphOperation : public Operation {
public:
    using ViewFunc = std::function<void(const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)>;
    using InferShapePreFunc =
        std::function<void(AsdOps::SVector<AsdOps::Tensor> &inTensors, AsdOps::SVector<AsdOps::Tensor> &outTensors)>;

    struct Node {
        std::shared_ptr<Operation> operation;
        AsdOps::SVector<uint64_t> inTensorIds;
        AsdOps::SVector<uint64_t> outTensorIds;
        AsdOps::SVector<ViewFunc> inTensorViewFuncs;
        InferShapePreFunc inferShapePreFunc;
        bool useVariantPackParam = false;
    };

    struct Graph {
        uint64_t inTensorSize = 0;
        uint64_t outTensorSize = 0;
        uint64_t intermediateTensorSize = 0;
        AsdOps::SVector<Node, 64> nodes;
        std::string ToString() const;
    };

    GraphOperation(const std::string &name);
    GraphOperation(const std::string &name, const Graph &opGraph);
    virtual ~GraphOperation();
    uint64_t GetInTensorCount() const override;
    uint64_t GetOutTensorCount() const override;

protected:
    Runner *CreateBestRunner() const override;
    RunnerBuilder *FindBestRunnerBuilder() const override;

protected:
    Graph opGraph_;
};
} // namespace AclTransformer
#endif