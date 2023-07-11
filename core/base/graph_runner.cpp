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
#include "acltransformer/graph_runner.h"
#include <sys/stat.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/utils/mem_allocation_solver/best_mem_allocation_solver.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "acltransformer/statistic.h"
#include "acltransformer/operation.h"

namespace AclTransformer {
const int ALIGN_INT = 32;

std::string GraphRunner::Graph::ToString() const
{
    std::stringstream ss;

    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " " << TensorUtil::AsdOpsTensorToString(inTensors.at(i))
           << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " "
           << TensorUtil::AsdOpsTensorToString(inTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto &node = nodes.at(i);
        ss << "node[" << i << "] opeation:" << node.operation.get()
           << ", operationName:" << (node.operation ? node.operation->GetName() : "null")
           << ", runner:" << node.runner.get() << ", runnerName:" << (node.runner ? node.runner->GetName() : "null")
           << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt << " " << TensorUtil::AsdOpsTensorToString(*tensorIt)
               << std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt << " " << TensorUtil::AsdOpsTensorToString(*tensorIt)
               << std::endl;
        }
    }
    return ss.str();
}

void GraphRunner::Graph::Init()
{
    InitTensorMaxNodeMap();
    InitTensorType();
}

void GraphRunner::Graph::InitTensorMaxNodeMap()
{
    tensorMaxNodeIdMap.clear();
    maxNodeIdTensorMap.clear();

    for (size_t i = 0; i < internalTensors.size(); ++i) {
        AsdOps::Tensor &internalTensor = internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
            auto &node = nodes.at(nodeId);
            for (auto inTensorIt : node.inTensors) {
                if (&internalTensor == inTensorIt) {
                    maxNodeId = nodeId;
                    dependNodeCount++;
                }
            }
        }
        tensorMaxNodeIdMap[&internalTensor] = maxNodeId;
        ASD_LOG_IF(dependNodeCount == 0, WARN) << "internal tensor[" << i << "] dependNodeCount is 0, graph wrong";
        maxNodeIdTensorMap[maxNodeId].insert(&internalTensor);
    }
}

void GraphRunner::Graph::InitTensorType()
{
    for (auto &node : nodes) {
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());

        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            if (IsInternalTensor(node.inTensors.at(i))) {
                node.inTensorTypes.at(i) = GraphRunner::INTERMEDIATE_TENSOR;
            }
        }

        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            if (IsInternalTensor(node.outTensors.at(i))) {
                node.outTensorTypes.at(i) = GraphRunner::INTERMEDIATE_TENSOR;
            }
        }
    }
}

bool GraphRunner::Graph::IsInternalTensor(const AsdOps::Tensor *tensor)
{
    for (auto &internalTensor : internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }

    return false;
}

GraphRunner::GraphRunner(const std::string &name) : Runner(name)
{
    memAllocatinSolver_.reset(new BestMemAllocationSolver());
}

GraphRunner::~GraphRunner() {}

GraphRunner::Graph &GraphRunner::GetGraph() { return runnerGraph_; }

AsdOps::Status GraphRunner::SetupImpl(const RunnerVariantPack &runnerVariantPack)
{
    ASD_LOG(INFO) << GetName() << " setup start";
    runnerGraph_.inTensors = runnerVariantPack.inTensors;
    runnerGraph_.outTensors = runnerVariantPack.outTensors;

    Reset();

    AsdOps::Status st = PreparseNodeVariantPack();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << GetName() << " setup fail, PreparseNodeVariantPack fail, error:" << st.Message();
        return st;
    }

    st = SetupAllRunners();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << GetName() << " setup fail, SetupAllRunners fail, error:" << st.Message();
        return st;
    }

    CalcTilingBufferSize();
    CalcIntermediateBufferSize();
    return AsdOps::Status::OkStatus();
}

uint64_t GraphRunner::GetTilingBufferSizeImpl() { return totalTilingBufferSize_; }

void GraphRunner::FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize)
{
    uint64_t tilingOffset = 0;
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runner->FillHostTilingBufferSize((char *)hostTilingBuffer + tilingOffset, tilingBufferSizes_.at(nodeId));
        tilingOffset += tilingBufferSizes_.at(nodeId);
    }
    ASD_LOG(INFO) << GetName() << " fill all node host tiling buffer";

    maxWorkspaceBufferSize_ = 0;
    workspaceBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerWorkspaceBufferSize = node.runner->GetWorkspaceBufferSize();
        workspaceBufferSizes_.at(nodeId) = runnerWorkspaceBufferSize;
        maxWorkspaceBufferSize_ = std::max(maxWorkspaceBufferSize_, runnerWorkspaceBufferSize);
        ASD_LOG(INFO) << GetName() << " node[" << nodeId << "] workspace buffer size:" << runnerWorkspaceBufferSize
                      << ", max:" << maxWorkspaceBufferSize_;
    }
    ASD_LOG(INFO) << GetName() << " max workspace buffer size:" << maxWorkspaceBufferSize_;
}

uint64_t GraphRunner::GetWorkspaceBufferSizeImpl() { return maxWorkspaceBufferSize_; }

uint64_t GraphRunner::GetIntermediateBufferSizeImpl()
{
    return selfIntermediateBufferSize_ + maxIntermediateBufferSize_;
}

AsdOps::Status GraphRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    ASD_LOG(INFO) << GetName() << " Execute start, runnerGraph_.nodes:" << runnerGraph_.nodes.size();

    if (handle.stream == nullptr) {
        ASD_LOG(ERROR) << GetName() << " Execute fail, handle.stream is null";
        return AsdOps::Status::FailStatus(1, "handle stream is null");
    }

    UpdateVariantPackBuffer(runnerVariantPack);
    UpdateVariantPackTensorData(runnerVariantPack);

    AsdOps::Status st = ExecuteAllRunner(handle, runnerVariantPack);
    if (!st.Ok()) {
        ASD_LOG(INFO) << GetName() << " execute fail";
        return st;
    }

    ASD_LOG(INFO) << GetName() << " execute success";
    return AsdOps::Status::OkStatus();
}

void GraphRunner::Reset()
{
    selfIntermediateBufferSize_ = 0;
    totalTilingBufferSize_ = 0;
    tilingBufferSizes_.clear();
    maxWorkspaceBufferSize_ = 0;
    workspaceBufferSizes_.clear();
    maxIntermediateBufferSize_ = 0;
    intermediateBufferSizes_.clear();
    memAllocatinSolver_->Reset();
    for (auto &tensor : runnerGraph_.internalTensors) {
        AsdOps::Tensor emptyTensor;
        tensor = emptyTensor;
    }
}

AsdOps::Status GraphRunner::PreparseNodeVariantPack()
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runnerVariantPack.inTensors.resize(node.inTensors.size());
        node.runnerVariantPack.outTensors.resize(node.outTensors.size());
        AsdOps::Status st = RunNodeInTensorViewFuncs(nodeId, node);
        if (!st.Ok()) {
            return st;
        }
        InferShapeNode(nodeId, node);
    }
    selfIntermediateBufferSize_ = memAllocatinSolver_->GetSize();
    ASD_LOG(INFO) << "GraphRunner MemAllocationSolver malloc size:" << memAllocatinSolver_->GetMallocSize()
                  << ", real size:" << memAllocatinSolver_->GetSize();
    return AsdOps::Status::OkStatus();
}

AsdOps::Status GraphRunner::RunNodeInTensorViewFuncs(size_t nodeId, GraphRunner::Node &node)
{
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        if (i < node.inTensorViewFuncs.size() && node.inTensorViewFuncs.at(i)) {
            AsdOps::Tensor viewTensor = *node.inTensors.at(i);
            viewTensor.desc.dims.clear();
            node.inTensorViewFuncs.at(i)(node.inTensors.at(i)->desc.dims, viewTensor.desc.dims);
            if (viewTensor.Numel() != node.inTensors.at(i)->Numel()) {
                ASD_LOG(ERROR) << GetName() << " node[" << nodeId
                               << "] invalid view func, viewTensor.Numel:" << viewTensor.Numel()
                               << ", tensor.Numel:" << node.inTensors.at(i)->Numel();
                return AsdOps::Status::FailStatus(1, "invalid view");
            }
            ASD_LOG(INFO) << GetName() << " node[" << nodeId << " view inTensor[" << i
                          << "], old:" << TensorUtil::AsdOpsDimsToString(node.inTensors.at(i)->desc.dims)
                          << ", new:" << TensorUtil::AsdOpsDimsToString(viewTensor.desc.dims);
            node.runnerVariantPack.inTensors.at(i) = viewTensor;
        } else {
            node.runnerVariantPack.inTensors.at(i) = *node.inTensors.at(i);
        }
    }
    return AsdOps::Status::OkStatus();
}

void GraphRunner::InferShapeNode(size_t nodeId, GraphRunner::Node &node)
{
    ASD_LOG(INFO) << GetName() << " node[" << nodeId << "] infer shape start";
    for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); ++i) {
        ASD_LOG(INFO) << GetName() << " " << node.runner->GetName() << " intensor[" << i << "] "
                      << TensorUtil::AsdOpsTensorToString(node.runnerVariantPack.inTensors.at(i));
    }
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    node.operation->InferShape(node.runnerVariantPack.inTensors, outTensorDescs);
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ASD_LOG(INFO) << GetName() << " " << node.runner->GetName() << " outTensorDescs[" << i << "] "
                      << TensorUtil::AsdOpsTensorDescToString(outTensorDescs.at(i));
    }
    ASD_LOG(INFO) << GetName() << " node[" << nodeId << "] infer shape end";

    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        AsdOps::Tensor *outTensor = node.outTensors.at(i);
        if (outTensor->data == nullptr) {
            outTensor->desc = outTensorDescs.at(i);
            outTensor->dataSize = TensorUtil::CalcTensorDataSize(*outTensor);
            outTensor->data = memAllocatinSolver_->Malloc(TensorUtil::AlignInt(outTensor->dataSize, ALIGN_INT));
            ASD_LOG(INFO) << GetName() << " " << node.runner->GetName()
                          << " MemAllocationSolver Malloc dataSize:" << outTensor->dataSize
                          << ", blockAddress:" << int64_t(outTensor->data);
        }
        node.runnerVariantPack.outTensors.at(i) = *outTensor;
        ASD_LOG(INFO) << GetName() << " " << node.runner->GetName() << " mem solve, outTensors[" << i << "] "
                      << TensorUtil::AsdOpsTensorToString(*outTensor);
    }

    auto it = runnerGraph_.maxNodeIdTensorMap.find(nodeId);
    if (it != runnerGraph_.maxNodeIdTensorMap.end()) {
        for (auto tensorIt : it->second) {
            ASD_LOG(INFO) << GetName() << " " << node.runner->GetName() << " free tensor:" << tensorIt;
            memAllocatinSolver_->Free((char *)tensorIt->data);
        }
    }
}

AsdOps::Status GraphRunner::SetupAllRunners()
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        AsdOps::Status st = node.runner->Setup(node.runnerVariantPack);
        if (!st.Ok()) {
            ASD_LOG(ERROR) << GetName() << " node[" << nodeId << "] setup fail, error:" << st.Message();
            return st;
        }
        ASD_LOG(INFO) << GetName() << " node[" << nodeId << "] setup success";
    }
    ASD_LOG(INFO) << GetName() << " setup all node success";
    return AsdOps::Status::OkStatus();
}

void GraphRunner::CalcTilingBufferSize()
{
    totalTilingBufferSize_ = 0;
    tilingBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerTilingBufferSize = node.runner->GetTilingBufferSize();
        ASD_LOG(INFO) << GetName() << " node[" << nodeId << "] tiling buffer size:" << runnerTilingBufferSize;
        totalTilingBufferSize_ += runnerTilingBufferSize;
        tilingBufferSizes_.at(nodeId) = runnerTilingBufferSize;
    }
    ASD_LOG(INFO) << GetName() << " total node tiling buffer size:" << totalTilingBufferSize_;
}

void GraphRunner::CalcIntermediateBufferSize()
{
    maxIntermediateBufferSize_ = 0;
    intermediateBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerIntermediateBufferSize = node.runner->GetIntermediateBufferSize();
        intermediateBufferSizes_.at(nodeId) = runnerIntermediateBufferSize;
        maxIntermediateBufferSize_ = std::max(maxIntermediateBufferSize_, runnerIntermediateBufferSize);
        ASD_LOG(INFO) << GetName() << " node[" << nodeId
                      << "] intermediate buffer size:" << runnerIntermediateBufferSize
                      << ", max:" << maxIntermediateBufferSize_;
    }
}

void GraphRunner::UpdateVariantPackBuffer(RunnerVariantPack &runnerVariantPack)
{
    ASD_LOG(INFO) << GetName() << " update runner variant pack's buffer start";
    if (totalTilingBufferSize_ > 0) {
        uint64_t offset = 0;
        for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
            auto &node = runnerGraph_.nodes.at(nodeId);
            node.runnerVariantPack.tilingBuffer = (char *)runnerVariantPack.tilingBuffer + offset;
            node.runnerVariantPack.tilingBufferSize = tilingBufferSizes_.at(nodeId);
            offset += tilingBufferSizes_.at(nodeId);
        }
    } else {
        ASD_LOG(WARN) << GetName() << " totalTilingBufferSize is 0, not update variantPack's tilingBuffer";
    }

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runnerVariantPack.workspaceBuffer = runnerVariantPack.workspaceBuffer;
        node.runnerVariantPack.workspaceBufferSize = workspaceBufferSizes_.at(nodeId);
    }

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runnerVariantPack.intermediateBuffer =
            (char *)runnerVariantPack.intermediateBuffer + selfIntermediateBufferSize_;
        node.runnerVariantPack.intermediateBufferSize = intermediateBufferSizes_.at(nodeId);
    }
    ASD_LOG(INFO) << GetName() << " update runner variant pack's buffer end";
}

void GraphRunner::UpdateVariantPackTensorData(RunnerVariantPack &runnerVariantPack)
{
    ASD_LOG(INFO) << GetName() << " update runner variant pack's tensor data start";
    char *selfIntermediateBuffer = static_cast<char *>(runnerVariantPack.intermediateBuffer);

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ASD_LOG(INFO) << GetName() << " update tensor.data node[" << nodeId << "]";
        for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); ++i) {
            auto &tensor = node.runnerVariantPack.inTensors.at(i);
            if (node.inTensorTypes.at(i) == GraphRunner::INTERMEDIATE_TENSOR) {
                tensor.data = selfIntermediateBuffer + (uint64_t)tensor.data;
                ASD_LOG(INFO) << GetName() << " update node[" << nodeId << "].intensors[" << i
                              << "] is internal, tensor.data:" << tensor.data;
            } else {
                tensor.data = node.inTensors.at(i)->data;
                ASD_LOG(INFO) << GetName() << " update node[" << nodeId
                              << "].intensor is not internal, tensor.data:" << tensor.data;
            }
        }
        for (size_t i = 0; i < node.runnerVariantPack.outTensors.size(); ++i) {
            auto &tensor = node.runnerVariantPack.outTensors.at(i);
            if (node.outTensorTypes.at(i) == GraphRunner::INTERMEDIATE_TENSOR) {
                tensor.data = selfIntermediateBuffer + (uint64_t)tensor.data;
                ASD_LOG(INFO) << GetName() << " update node[" << nodeId << "].outtensor[" << i
                              << "] is internal, tensor.data:" << tensor.data;
            } else {
                tensor.data = node.outTensors.at(i)->data;
                ASD_LOG(INFO) << GetName() << " update node[" << nodeId << "].outtensor[" << i
                              << "] is not internal, tensor.data:" << tensor.data;
            }
        }
    }
    ASD_LOG(INFO) << GetName() << " update runner variant pack's tensor data end";
}

AsdOps::Status GraphRunner::ExecuteAllRunner(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ASD_LOG(INFO) << GetName() << " node[" << nodeId << "] execute start, runner name:" << node.runner->GetName()
                      << ", variantPack:\n"
                      << node.runnerVariantPack.ToString();

        AsdOps::Status st = node.runner->Execute(handle, node.runnerVariantPack);
        if (!st.Ok()) {
            ASD_LOG(ERROR) << GetName() << " node[" << nodeId
                           << "] execute fail, runner name:" << node.runner->GetName();
            return st;
        }

        if (AsdOps::GetSingleton<Config>().IsStreamSyncEveryRunnerEnable()) {
            AsdOps::Timer timer;
            int ret = AsdRtStreamSynchronize(handle.stream);
            AsdOps::GetSingleton<Statistic>().syclTime += timer.ElapsedMicroSecond();
            ASD_LOG_IF(ret != 0, ERROR) << GetName() << " node[" << nodeId << "] stream sync fail, ret:" << ret;
        }

        if (AsdOps::GetSingleton<Config>().IsSaveTensor()) {
            AsdRtStreamSynchronize(handle.stream);
            std::string dirPath = Config::GetSaveTensorDir() + "/" + GetName() + "/" + std::to_string(nodeId) + "_" +
                                  node.runner->GetName();
            TensorUtil::SaveVariantPack(handle, node.runnerVariantPack, dirPath);
            ASD_LOG(INFO) << GetName() << " node[" << nodeId << "] save runner variant pack, dir:" << dirPath;
        }
    }

    if (AsdOps::GetSingleton<Config>().IsStreamSyncEveryPlanEnable()) {
        int ret = AsdRtStreamSynchronize(handle.stream);
        ASD_LOG_IF(ret != 0, ERROR) << GetName() << " stream sync  fail, ret:" << ret;
    }

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer