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
#include "acltransformer/runner/graph_runner.h"
#include <sys/stat.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/utils/mem_allocation_solver/best_mem_allocation_solver.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "acltransformer/statistic.h"

namespace AclTransformer {
GraphRunner::GraphRunner(const std::string &name) : Runner(name)
{
    memAllocatinSolver_.reset(new BestMemAllocationSolver());
}

GraphRunner::~GraphRunner() {}

AsdOps::Status GraphRunner::SetupImpl(const VariantPack &variantPack)
{
    ASD_LOG(INFO) << name_ << " setup start";
    runnerGraph_.inTensors = variantPack.inTensors;
    runnerGraph_.outTensors = variantPack.outTensors;

    Reset();

    AsdOps::Status st = PreparseNodeVariantPack();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " setup fail, PreparseNodeVariantPack fail, error:" << st.Message();
        return st;
    }

    st = SetupAllRunners();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " setup fail, SetupAllRunners fail, error:" << st.Message();
        return st;
    }

    CalcTilingBufferSize();
    CalcIntermediateBufferSize();

    return AsdOps::Status::OkStatus();
}

void GraphRunner::CalcTilingBufferSize()
{
    totalTilingBufferSize_ = 0;
    tilingBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerTilingBufferSize = node.runner->GetTilingBufferSize();
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] tiling buffer size:" << runnerTilingBufferSize;
        totalTilingBufferSize_ += runnerTilingBufferSize;
        tilingBufferSizes_.at(nodeId) = runnerTilingBufferSize;
    }
    ASD_LOG(INFO) << name_ << " total node tiling buffer size:" << totalTilingBufferSize_;
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
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] intermediate buffer size:" << runnerIntermediateBufferSize
                      << ", max:" << maxIntermediateBufferSize_;
    }
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
    ASD_LOG(INFO) << name_ << " fill all node host tiling buffer";

    maxWorkspaceBufferSize_ = 0;
    workspaceBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerWorkspaceBufferSize = node.runner->GetWorkspaceBufferSize();
        workspaceBufferSizes_.at(nodeId) = runnerWorkspaceBufferSize;
        maxWorkspaceBufferSize_ = std::max(maxWorkspaceBufferSize_, runnerWorkspaceBufferSize);
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] workspace buffer size:" << runnerWorkspaceBufferSize
                      << ", max:" << maxWorkspaceBufferSize_;
    }
    ASD_LOG(INFO) << name_ << " max workspace buffer size:" << maxWorkspaceBufferSize_;
}

uint64_t GraphRunner::GetWorkspaceBufferSizeImpl() { return maxWorkspaceBufferSize_; }

uint64_t GraphRunner::GetIntermediateBufferSizeImpl()
{
    return selfIntermediateBufferSize_ + maxIntermediateBufferSize_;
}

AsdOps::Status GraphRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    ASD_LOG(INFO) << name_ << " Execute start, runnerGraph_.nodes:" << runnerGraph_.nodes.size();

    if (handle.stream == nullptr) {
        ASD_LOG(ERROR) << name_ << " Execute fail, handle.stream is null";
        return AsdOps::Status::FailStatus(1, "handle stream is null");
    }

    UpdateVariantPackBuffer(variantPack);
    UpdateVariantPackTensorData(variantPack);

    AsdOps::Status st = ExecuteAllRunner(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(INFO) << name_ << " execute fail";
        return st;
    }

    ASD_LOG(INFO) << name_ << " execute success";
    return AsdOps::Status::OkStatus();
}

AsdOps::Status GraphRunner::ExecuteAllRunner(Handle &handle, VariantPack &variantPack)
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] execute start, runner name:" << node.runner->GetName()
                      << ", variantPack:\n"
                      << node.variantPack.ToString();

        AsdOps::Status st = node.runner->Execute(handle, node.variantPack);
        if (!st.Ok()) {
            ASD_LOG(ERROR) << name_ << " node[" << nodeId << "] execute fail, runner name:" << node.runner->GetName();
            return st;
        }

        if (AsdOps::GetSingleton<Config>().IsStreamSyncEveryRunnerEnable()) {
            AsdOps::Timer timer;
            int ret = AsdRtStreamSynchronize(handle.stream);
            AsdOps::GetSingleton<Statistic>().syclTime += timer.ElapsedMicroSecond();
            ASD_LOG_IF(ret != 0, ERROR) << name_ << " node[" << nodeId << "] stream sync fail, ret:" << ret;
        }

        if (AsdOps::GetSingleton<Config>().IsSaveTensor()) {
            AsdRtStreamSynchronize(handle.stream);
            std::string dirPath =
                Config::GetSaveTensorDir() + "/" + name_ + "/" + std::to_string(nodeId) + "_" + node.runner->GetName();
            TensorUtil::SaveVariantPack(handle, node.variantPack, dirPath);
            ASD_LOG(INFO) << name_ << " node[" << nodeId << "] save runner variant pack, dir:" << dirPath;
        }
    }

    if (AsdOps::GetSingleton<Config>().IsStreamSyncEveryPlanEnable()) {
        int ret = AsdRtStreamSynchronize(handle.stream);
        ASD_LOG_IF(ret != 0, ERROR) << name_ << " stream sync  fail, ret:" << ret;
    }

    return AsdOps::Status::OkStatus();
}

void GraphRunner::UpdateVariantPackBuffer(VariantPack &variantPack)
{
    ASD_LOG(INFO) << name_ << " update runner variant pack's buffer start";
    if (totalTilingBufferSize_ > 0) {
        uint64_t offset = 0;
        for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
            auto &node = runnerGraph_.nodes.at(nodeId);
            node.variantPack.tilingBuffer = (char *)variantPack.tilingBuffer + offset;
            node.variantPack.tilingBufferSize = tilingBufferSizes_.at(nodeId);
            offset += tilingBufferSizes_.at(nodeId);
        }
    } else {
        ASD_LOG(WARN) << name_ << " totalTilingBufferSize is 0, not update variantPack's tilingBuffer";
    }

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.variantPack.workspaceBuffer = variantPack.workspaceBuffer;
        node.variantPack.workspaceBufferSize = workspaceBufferSizes_.at(nodeId);
    }

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.variantPack.intermediateBuffer = (char *)variantPack.intermediateBuffer + selfIntermediateBufferSize_;
        node.variantPack.intermediateBufferSize = intermediateBufferSizes_.at(nodeId);
    }
    ASD_LOG(INFO) << name_ << " update runner variant pack's buffer end";
}

void GraphRunner::UpdateVariantPackTensorData(VariantPack &variantPack)
{
    ASD_LOG(INFO) << name_ << " update runner variant pack's tensor data start";
    char *selfIntermediateBuffer = static_cast<char *>(variantPack.intermediateBuffer);

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ASD_LOG(INFO) << name_ << " update tensor.data node[" << nodeId << "]";
        for (size_t i = 0; i < node.variantPack.inTensors.size(); ++i) {
            auto &tensor = node.variantPack.inTensors.at(i);
            if (IsInternalTensor(node.inTensors.at(i))) {
                tensor.data = selfIntermediateBuffer + (uint64_t)tensor.data;
                ASD_LOG(INFO) << name_ << " update node[" << nodeId << "].intensors[" << i
                              << "] is internal, tensor.data:" << tensor.data;
            } else {
                tensor.data = GetInOrOutTensorData(node.inTensors.at(i), variantPack);

                ASD_LOG(INFO) << name_ << " update node[" << nodeId << "].intensor is not internal";
            }
        }
        for (size_t i = 0; i < node.variantPack.outTensors.size(); ++i) {
            auto &tensor = node.variantPack.outTensors.at(i);
            if (IsInternalTensor(node.outTensors.at(i))) {
                tensor.data = selfIntermediateBuffer + (uint64_t)tensor.data;
                ASD_LOG(INFO) << name_ << " update node[" << nodeId << "].outtensor[" << i
                              << "] is internal, tensor.data:" << tensor.data;
            } else {
                tensor.data = GetInOrOutTensorData(node.outTensors.at(i), variantPack);
                ASD_LOG(INFO) << name_ << " update node[" << nodeId << "].outtensor[" << i << "] is not internal";
            }
        }
    }
    ASD_LOG(INFO) << name_ << " update runner variant pack's tensor data end";
}

void *GraphRunner::GetInOrOutTensorData(AsdOps::Tensor *tensor, const VariantPack &variantPack)
{
    int64_t tensorId = GetInTensorId(tensor);
    if (tensorId != -1) {
        return variantPack.inTensors.at(tensorId).data;
    }

    tensorId = GetOutTensorId(tensor);
    if (tensorId != -1) {
        return variantPack.outTensors.at(tensorId).data;
    }
    return nullptr;
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

bool GraphRunner::IsInternalTensor(const AsdOps::Tensor *tensor)
{
    for (auto &internalTensor : runnerGraph_.internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }

    return false;
}

int64_t GraphRunner::GetInTensorId(const AsdOps::Tensor *tensor)
{
    for (size_t i = 0; i < runnerGraph_.inTensors.size(); ++i) {
        if (&runnerGraph_.inTensors.at(i) == tensor) {
            return i;
        }
    }
    return -1;
}

int64_t GraphRunner::GetOutTensorId(const AsdOps::Tensor *tensor)
{
    for (size_t i = 0; i < runnerGraph_.outTensors.size(); ++i) {
        if (&runnerGraph_.outTensors.at(i) == tensor) {
            return i;
        }
    }
    return -1;
}

AsdOps::Status GraphRunner::PreparseNodeVariantPack()
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.variantPack.inTensors.resize(node.inTensors.size());
        node.variantPack.outTensors.resize(node.outTensors.size());
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

AsdOps::Status GraphRunner::RunNodeInTensorViewFuncs(size_t nodeId, RunnerGraphNode &node)
{
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        if (i < node.inTensorViewFuncs.size() && node.inTensorViewFuncs.at(i)) {
            AsdOps::Tensor viewTensor = *node.inTensors.at(i);
            viewTensor.desc.dims.clear();
            node.inTensorViewFuncs.at(i)(node.inTensors.at(i)->desc.dims, viewTensor.desc.dims);
            if (viewTensor.Numel() != node.inTensors.at(i)->Numel()) {
                ASD_LOG(ERROR) << name_ << " node[" << nodeId
                               << "] invalid view func, viewTensor.Numel:" << viewTensor.Numel()
                               << ", tensor.Numel:" << node.inTensors.at(i)->Numel();
                return AsdOps::Status::FailStatus(1, "invalid view");
            }
            ASD_LOG(INFO) << name_ << " node[" << nodeId << " view inTensor[" << i
                          << "], old:" << TensorUtil::AsdOpsDimsToString(node.inTensors.at(i)->desc.dims)
                          << ", new:" << TensorUtil::AsdOpsDimsToString(viewTensor.desc.dims);
            node.variantPack.inTensors.at(i) = viewTensor;
        } else {
            node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
        }
    }
    return AsdOps::Status::OkStatus();
}

void GraphRunner::InferShapeNode(size_t nodeId, RunnerGraphNode &node)
{
    ASD_LOG(INFO) << name_ << " node[" << nodeId << "] infer shape start";
    for (size_t i = 0; i < node.variantPack.inTensors.size(); ++i) {
        ASD_LOG(INFO) << name_ << " " << node.runner->GetName() << " intensor[" << i << "] "
                      << TensorUtil::AsdOpsTensorToString(node.variantPack.inTensors.at(i));
    }
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    node.operation->InferShape(node.variantPack.inTensors, outTensorDescs);
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ASD_LOG(INFO) << name_ << " " << node.runner->GetName() << " outTensorDescs[" << i << "] "
                      << TensorUtil::AsdOpsTensorDescToString(outTensorDescs.at(i));
    }
    ASD_LOG(INFO) << name_ << " node[" << nodeId << "] infer shape end";

    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        AsdOps::Tensor *outTensor = node.outTensors.at(i);
        if (outTensor->data == nullptr) {
            outTensor->desc = outTensorDescs.at(i);
            outTensor->dataSize = TensorUtil::CalcTensorDataSize(*outTensor);
            outTensor->data = memAllocatinSolver_->Malloc(TensorUtil::AlignInt(outTensor->dataSize, 32));
            ASD_LOG(INFO) << name_ << " " << node.runner->GetName()
                          << " MemAllocationSolver Malloc dataSize:" << outTensor->dataSize
                          << ", blockAddress:" << int64_t(outTensor->data);
        }
        node.variantPack.outTensors.at(i) = *outTensor;
        ASD_LOG(INFO) << name_ << " " << node.runner->GetName() << " mem solve, outTensors[" << i << "] "
                      << TensorUtil::AsdOpsTensorToString(*outTensor);
    }

    auto it = runnerGraph_.maxNodeIdTensorMap.find(nodeId);
    if (it != runnerGraph_.maxNodeIdTensorMap.end()) {
        for (auto tensorIt : it->second) {
            ASD_LOG(INFO) << name_ << " " << node.runner->GetName() << " free tensor:" << tensorIt;
            memAllocatinSolver_->Free((char *)tensorIt->data);
        }
    }
}

AsdOps::Status GraphRunner::SetupAllRunners()
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        AsdOps::Status st = node.runner->Setup(node.variantPack);
        if (!st.Ok()) {
            ASD_LOG(ERROR) << name_ << " node[" << nodeId << "] setup fail, error:" << st.Message();
            return st;
        }
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] setup success";
    }
    ASD_LOG(INFO) << name_ << " setup all node success";
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer