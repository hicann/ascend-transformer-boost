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
#include "acltransformer/ops/linear_parallel_operation.h"
#include "acltransformer/ops/matmul_operation.h"
#include "acltransformer/ops/all_reduce_operation.h"
#include "acltransformer/ops/add_operation.h"

namespace AclTransformer {
enum LinearParallelTensorId {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_BIAS,
    OUT_LINEAROUT,
    INTERMIDATE_MATMULOUT,
    INTERMIDATE_ALLREDUCEOUT,
};

static uint64_t IN_TENSOR_COUNT = 3;
static uint64_t OUT_TENSOR_COUNT = 1;
static uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static uint64_t NODE_COUNT = 3;

LinearParallelOperation::LinearParallelOperation(const LinearParallelParam &param)
    : GraphOperation("LinearParallelOperation"), param_(param)
{
    if (param_.bias != "None") {
        opGraph_.inTensorSize = IN_TENSOR_COUNT;
        opGraph_.outTensorSize = OUT_TENSOR_COUNT;
        opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
        opGraph_.nodes.resize(NODE_COUNT);

        size_t nodeId = 0;
        GraphOperation::Node &matmulNode = opGraph_.nodes.at(nodeId++);
        GraphOperation::Node &allReduceNode = opGraph_.nodes.at(nodeId++);
        GraphOperation::Node &addNode = opGraph_.nodes.at(nodeId++);

        matmulNode.operation.reset(new AclTransformer::MatmulOperation({false, !param_.transWeight}));
        matmulNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
        matmulNode.outTensorIds = {INTERMIDATE_MATMULOUT};
        matmulNode.inTensorViewFuncs.resize(matmulNode.inTensorIds.size());
        matmulNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                             AsdOps::SVector<int64_t> &newDims) {
            // weight dimension suported: only 2
            int64_t dim0 = 1;
            for (size_t i = 0; i < oldDims.size() - 1; i++) {
                dim0 *= oldDims.at(i);
            }
            int64_t dim1 = oldDims.at(oldDims.size() - 1);
            newDims = {dim0, dim1};
        };

        allReduceNode.operation.reset(
            new AclTransformer::AllReduceOperation({param_.rank, param_.rankSize, param_.rankRoot, "sum", param_.backend, param_.useCommExt, param_.commExt}));
        allReduceNode.inTensorIds = {INTERMIDATE_MATMULOUT};
        allReduceNode.outTensorIds = {INTERMIDATE_ALLREDUCEOUT};

        addNode.operation.reset(new AclTransformer::AddOperation({1}));
        addNode.inTensorIds = {INTERMIDATE_ALLREDUCEOUT, IN_BIAS};
        addNode.outTensorIds = {OUT_LINEAROUT};
    } else {
        opGraph_.inTensorSize = IN_TENSOR_COUNT - 1;
        opGraph_.outTensorSize = OUT_TENSOR_COUNT;
        opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT - 1;
        opGraph_.nodes.resize(NODE_COUNT - 1);

        size_t nodeId = 0;
        GraphOperation::Node &matmulNode = opGraph_.nodes.at(nodeId++);
        GraphOperation::Node &allReduceNode = opGraph_.nodes.at(nodeId++);

        matmulNode.operation.reset(new AclTransformer::MatmulOperation({false, !param_.transWeight}));
        matmulNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
        matmulNode.outTensorIds = {INTERMIDATE_MATMULOUT - 1};
        matmulNode.inTensorViewFuncs.resize(matmulNode.inTensorIds.size());
        matmulNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                             AsdOps::SVector<int64_t> &newDims) {
            // weight dimension suported: only 2
            int64_t dim0 = 1;
            for (size_t i = 0; i < oldDims.size() - 1; i++) {
                dim0 *= oldDims.at(i);
            }
            int64_t dim1 = oldDims.at(oldDims.size() - 1);
            newDims = {dim0, dim1};
        };

        allReduceNode.operation.reset(
            new AclTransformer::AllReduceOperation({param_.rank, param_.rankSize, param_.rankRoot, "sum", param_.backend, param_.useCommExt, param_.commExt}));
        allReduceNode.inTensorIds = {INTERMIDATE_MATMULOUT - 1};
        allReduceNode.outTensorIds = {OUT_LINEAROUT - 1};
    }
}

LinearParallelOperation::~LinearParallelOperation() {}

uint64_t LinearParallelOperation::GetInTensorCount() const
{
    if (param_.bias != "None") {
        return IN_TENSOR_COUNT;
    } else {
        return IN_TENSOR_COUNT - 1;
    }
}

uint64_t LinearParallelOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status LinearParallelOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                       AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    outTensorDescs.at(0).dims = inTensors.at(0).desc.dims;
    auto outDimSize = outTensorDescs.at(0).dims.size();
    if (!param_.transWeight) {
        outTensorDescs.at(0).dims[outDimSize - 1] = inTensors.at(1).desc.dims[0];
    } else {
        outTensorDescs.at(0).dims[outDimSize - 1] = inTensors.at(1).desc.dims[1];
    }

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer