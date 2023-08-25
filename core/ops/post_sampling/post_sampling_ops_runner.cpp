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
#include "post_sampling_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PostSamplingOpsRunner::PostSamplingOpsRunner(const PostSamplingParam &param) : OpsRunner("PostSamplingOpsRunner", RUNNER_TYPE_POST_SAMPLING), param_(param)
{
    ASD_LOG(INFO) << "PostSamplingOpsRunner::PostSamplingOpsRunner called";

    const std::size_t inTensorSize = 3;
    const std::size_t outTensorSize = 1;
    const std::size_t internalTensorSize = 10;
    const std::size_t nodeSize = 10;

    kernelGraph_.inTensors.resize(inTensorSize);
    size_t inTensorId = 0;
    AsdOps::Tensor &logits = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &zeroTensor = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &topPTensor = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &resultTokenTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.internalTensors.resize(internalTensorSize);
    size_t internalTensorId = 0;
    AsdOps::Tensor &sortedTopKLogits = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &sortedTopKIndices = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &softMaxOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &cumulativeProbs = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &greaterOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &sliceOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &indicesToRemove = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &newLogits = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &newSoftmaxOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &pred = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(nodeSize);
    size_t nodeId = 0;
    auto &topKode = kernelGraph_.nodes[nodeId++];
    auto &firstSoftmaxNode = kernelGraph_.nodes[nodeId++];
    auto &cunsumNode = kernelGraph_.nodes[nodeId++];
    auto &greaterNode = kernelGraph_.nodes[nodeId++];
    auto &sliceNode = kernelGraph_.nodes[nodeId++];
    auto &concatNode = kernelGraph_.nodes[nodeId++];
    auto &maskfillNode = kernelGraph_.nodes[nodeId++];
    auto &secondSoftmaxNode = kernelGraph_.nodes[nodeId++];
    auto &multinomialNode = kernelGraph_.nodes[nodeId++];
    auto &gatherNode = kernelGraph_.nodes[nodeId++];

    ViewFunc Squeeze = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1)};
    };

    topKode.opDesc = {0, "SortOperation", AsdOps::OpParam::Sort({param_.topK})};
    topKode.inTensors = {&logits};
    topKode.outTensors = {&sortedTopKLogits, &sortedTopKIndices};
    topKode.inTensorViewFuncs.resize(topKode.inTensors.size());
    topKode.inTensorViewFuncs.at(0) = Squeeze;

    firstSoftmaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    firstSoftmaxNode.inTensors = {&sortedTopKLogits};
    firstSoftmaxNode.outTensors = {&softMaxOut};

    cunsumNode.opDesc = {0, "CumsumOperation", AsdOps::OpParam::Cumsum({AsdOps::OpParam::Cumsum::CUMSUM, {0}, false, false})};
    cunsumNode.inTensors = {&softMaxOut};
    cunsumNode.outTensors = {&cumulativeProbs};

    greaterNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_GREATER})};
    greaterNode.inTensors = {&cumulativeProbs, &topPTensor};
    greaterNode.outTensors = {&greaterOut};

    sliceNode.opDesc = {0, "SliceOperation", AsdOps::OpParam::Slice({AsdOps::OpParam::Slice::SLICE, {0}, {param_.topK - 1}})};
    sliceNode.inTensors = {&greaterOut};
    sliceNode.outTensors = {&sliceOut};

    concatNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    concatNode.inTensors = {&zeroTensor, &sliceOut};
    concatNode.outTensors = {&indicesToRemove};

    const float maskValue = -65504.0;
    maskfillNode.opDesc = {0,"BroadcastOperation",
                            AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MASKEDFILL, {maskValue}})};
    maskfillNode.inTensors = {&sortedTopKLogits, &indicesToRemove};
    maskfillNode.outTensors = {&newLogits};

    secondSoftmaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    secondSoftmaxNode.inTensors = {&newLogits};
    secondSoftmaxNode.outTensors = {&newSoftmaxOut};

    multinomialNode.opDesc = {0, "multinomialOperation", AsdOps::OpParam::Multinomial({AsdOps::OpParam::Multinomial::MULTINOMIAL, 1})};
    multinomialNode.inTensors = {&newSoftmaxOut};
    multinomialNode.outTensors = {&pred};

    gatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    gatherNode.inTensors = {&sortedTopKIndices, &pred};
    gatherNode.outTensors = {&resultTokenTensor};
}

PostSamplingOpsRunner::~PostSamplingOpsRunner() {}
} // namespace AclTransformer
