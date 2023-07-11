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
#include "acltransformer/plan.h"
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
std::string RunnerGraph::ToString() const
{
    std::stringstream ss;
    return ss.str();
}

Plan::Plan()
{
    memAllocatinSolver_.reset(new BestMemAllocationSolver());
    totalHostTilingBuffer_.reserve(1024 * 1024);
}

Plan::~Plan() {}

AsdOps::Status Plan::Setup(Handle handle, const VariantPack &variantPack)
{
    ASD_LOG(INFO) << name_ << " setup start";
    runnerGraph_.inTensors = variantPack.inTensors;
    runnerGraph_.outTensors = variantPack.outTensors;

    Reset();

    AsdOps::Status st = PreparseNodeRunnerVariantPack();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " setup fail, PreparseNodeRunnerVariantPack fail, error:" << st.Message();
        return st;
    }

    st = SetupAllRunners();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " setup fail, SetupAllRunners fail, error:" << st.Message();
        return st;
    }

    ASD_LOG(INFO) << name_ << " setup success, totalTilingBufferSize:" << totalTilingBufferSize_
                  << ", maxWorkspaceBufferSize:" << maxWorkspaceBufferSize_
                  << ", maxIntermediateBufferSize:" << maxIntermediateBufferSize_
                  << ", selfIntermediateBufferSize:" << selfIntermediateBufferSize_;
    return AsdOps::Status::OkStatus();
}

uint64_t Plan::GetWorkspaceSize()
{
    return totalTilingBufferSize_ + maxWorkspaceBufferSize_ + maxIntermediateBufferSize_ + selfIntermediateBufferSize_;
}

AsdOps::Status Plan::Execute(Handle handle, VariantPack &variantPack)
{
    ASD_LOG(INFO) << name_ << " Execute start, runnerGraph_.nodes:" << runnerGraph_.nodes.size();

    if (handle.stream == nullptr) {
        ASD_LOG(ERROR) << name_ << " Execute fail, handle.stream is null";
        return AsdOps::Status::FailStatus(1, "handle stream is null");
    }
    runnerGraph_.inTensors = variantPack.inTensors;
    runnerGraph_.outTensors = variantPack.outTensors;

    AsdOps::Status st = CopyHostTilingToDevice(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(INFO) << name_ << " execute fail";
        return st;
    }

    UpdateRunnerVariantPackBuffer(variantPack);
    UpdateRunnerVariantPackTensorData(variantPack);
    st = ExecuteAllRunner(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(INFO) << name_ << " execute fail";
        return st;
    }

    ASD_LOG(INFO) << name_ << " execute success";
    return AsdOps::Status::OkStatus();
}

AsdOps::Status Plan::ExecuteAllRunner(Handle &handle, VariantPack &variantPack)
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] execute start, runner name:" << node.runner->GetName()
                      << ", variantPack:\n"
                      << node.runnerVariantPack.ToString();

        AsdOps::Status st = node.runner->Execute(handle, node.runnerVariantPack);
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
            TensorUtil::SaveVariantPack(handle, node.runnerVariantPack, dirPath);
            ASD_LOG(INFO) << name_ << " node[" << nodeId << "] save runner variant pack, dir:" << dirPath;
        }
    }

    if (AsdOps::GetSingleton<Config>().IsStreamSyncEveryPlanEnable()) {
        int ret = AsdRtStreamSynchronize(handle.stream);
        ASD_LOG_IF(ret != 0, ERROR) << name_ << " stream sync  fail, ret:" << ret;
    }

    return AsdOps::Status::OkStatus();
}

AsdOps::Status Plan::CopyHostTilingToDevice(Handle handle, VariantPack &variantPack)
{
    if (totalTilingBufferSize_ > 0) {
        char *totalTilingBuffer = static_cast<char *>(variantPack.workspace);
        ASD_LOG(INFO) << name_ << " copy host tiling to device start, totalTilingBufferSize:" << totalTilingBufferSize_;
        AsdOps::Timer timer;
        int ret = AsdRtMemCopyAsync(totalTilingBuffer, totalTilingBufferSize_, totalHostTilingBuffer_.data(),
                                    totalTilingBufferSize_, ASDRT_MEMCOPY_HOST_TO_DEVICE, handle.stream);
        AsdOps::GetSingleton<Statistic>().tillingCopyTime = timer.ElapsedMicroSecond();
        if (ret != 0) {
            ASD_LOG(ERROR) << name_ << " copy host tiling to device fail, ret:" << ret;
            return AsdOps::Status::FailStatus(1, "copy host tiling to device fail");
        }
    }
    return AsdOps::Status::OkStatus();
}

void Plan::UpdateRunnerVariantPackBuffer(VariantPack &variantPack)
{
    ASD_LOG(INFO) << name_ << " update runner variant pack's buffer start";
    if (totalTilingBufferSize_ > 0) {
        char *totalTilingBuffer = static_cast<char *>(variantPack.workspace);
        uint64_t offset = 0;
        for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
            auto &node = runnerGraph_.nodes.at(nodeId);
            node.runnerVariantPack.tilingBuffer = totalTilingBuffer + offset;
            node.runnerVariantPack.tilingBufferSize = tilingBufferSizes_.at(nodeId);
            offset += tilingBufferSizes_.at(nodeId);
        }
    } else {
        ASD_LOG(WARN) << name_ << " totalTilingBufferSize is 0, not update runnerVariantPack's tilingBuffer";
    }

    char *totalWorkspaceBufer = static_cast<char *>(variantPack.workspace) + totalTilingBufferSize_;
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runnerVariantPack.workspaceBuffer = totalWorkspaceBufer;
        node.runnerVariantPack.workspaceBufferSize = workspaceBufferSizes_.at(nodeId);
    }

    char *totalIntermediateBuffer =
        static_cast<char *>(variantPack.workspace) + totalTilingBufferSize_ + maxWorkspaceBufferSize_;
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runnerVariantPack.intermediateBuffer = totalIntermediateBuffer;
        node.runnerVariantPack.intermediateBufferSize = intermediateBufferSizes_.at(nodeId);
    }
    ASD_LOG(INFO) << name_ << " update runner variant pack's buffer end";
}

void Plan::UpdateRunnerVariantPackTensorData(VariantPack &variantPack)
{
    ASD_LOG(INFO) << name_ << " update runner variant pack's tensor data start";
    char *selfIntermediateBuffer = static_cast<char *>(variantPack.workspace) + totalTilingBufferSize_ +
                                   maxWorkspaceBufferSize_ + maxIntermediateBufferSize_;

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ASD_LOG(INFO) << name_ << " update tensor.data node[" << nodeId << "]";
        for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); ++i) {
            auto &tensor = node.runnerVariantPack.inTensors.at(i);
            if (node.inTensorTypes.at(i) == INTERMEDIATE_TENSOR) {
                tensor.data = selfIntermediateBuffer + (uint64_t)tensor.data;
                ASD_LOG(INFO) << name_ << " update node[" << nodeId << "].intensors[" << i
                              << "] is internal, tensor.data:" << tensor.data;
            } else {
                tensor.data = node.inTensors.at(i)->data;
                ASD_LOG(INFO) << name_ << " update node[" << nodeId << "].intensor is not internal";
            }
        }
        for (size_t i = 0; i < node.runnerVariantPack.outTensors.size(); ++i) {
            auto &tensor = node.runnerVariantPack.outTensors.at(i);
            if (node.outTensorTypes.at(i) == INTERMEDIATE_TENSOR) {
                tensor.data = selfIntermediateBuffer + (uint64_t)tensor.data;
                ASD_LOG(INFO) << name_ << " update node[" << nodeId << "].outtensor[" << i
                              << "] is internal, tensor.data:" << tensor.data;
            } else {
                tensor.data = node.outTensors.at(i)->data;
                ASD_LOG(INFO) << name_ << " update node[" << nodeId << "].outtensor[" << i << "] is not internal";
            }
        }
    }
    ASD_LOG(INFO) << name_ << " update runner variant pack's tensor data end";
}

void Plan::Reset()
{
    selfIntermediateBufferSize_ = 0;
    totalTilingBufferSize_ = 0;
    tilingBufferSizes_.clear();
    totalHostTilingBuffer_.clear();
    maxWorkspaceBufferSize_ = 0;
    workspaceBufferSizes_.clear();
    maxIntermediateBufferSize_ = 0;
    intermediateBufferSizes_.clear();
    memAllocatinSolver_->Reset();
    for (size_t i = 0; i < runnerGraph_.internalTensors.size(); i++) {
        AsdOps::Tensor emptyTensor;
        runnerGraph_.internalTensors.at(i) = emptyTensor;
    }
}

bool Plan::IsInternalTensor(const AsdOps::Tensor *tensor)
{
    for (auto &internalTensor : runnerGraph_.internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }

    return false;
}

int64_t Plan::GetInTensorId(const AsdOps::Tensor *tensor)
{
    for (size_t i = 0; i < runnerGraph_.inTensors.size(); ++i) {
        if (&runnerGraph_.inTensors.at(i) == tensor) {
            return i;
        }
    }
    return -1;
}

int64_t Plan::GetOutTensorId(const AsdOps::Tensor *tensor)
{
    for (size_t i = 0; i < runnerGraph_.outTensors.size(); ++i) {
        if (&runnerGraph_.outTensors.at(i) == tensor) {
            return i;
        }
    }
    return -1;
}

void Plan::InitTensorMaxNodeMap()
{
    tensorMaxNodeIdMap_.clear();
    maxNodeIdTensorMap_.clear();
    for (size_t i = 0; i < runnerGraph_.internalTensors.size(); ++i) {
        AsdOps::Tensor &internalTensor = runnerGraph_.internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
            auto &node = runnerGraph_.nodes.at(nodeId);
            for (auto inTensorIt : node.inTensors) {
                if (&internalTensor == inTensorIt) {
                    maxNodeId = nodeId;
                    dependNodeCount++;
                }
            }
        }
        tensorMaxNodeIdMap_[&internalTensor] = maxNodeId;
        ASD_LOG(INFO) << name_ << " internal tensor[" << i << "] maxNodeId:" << maxNodeId
                      << ", dependNodeCount:" << dependNodeCount;
        ASD_LOG_IF(dependNodeCount == 0, ERROR) << "internal tensor[" << i << "] dependNodeCount is 0, graph wrong";
        maxNodeIdTensorMap_[maxNodeId].insert(&internalTensor);
    }
}

void Plan::InitTensorType()
{
    for (auto &node : runnerGraph_.nodes) {
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            node.inTensorTypes.at(i) =
                IsInternalTensor(node.inTensors.at(i)) ? INTERMEDIATE_TENSOR : NOT_INTERMEDIATE_TENSOR;
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            node.outTensorTypes.at(i) =
                IsInternalTensor(node.outTensors.at(i)) ? INTERMEDIATE_TENSOR : NOT_INTERMEDIATE_TENSOR;
        }
    }
}

AsdOps::Status Plan::PreparseNodeRunnerVariantPack()
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
    ASD_LOG(INFO) << "Plan MemAllocationSolver malloc size:" << memAllocatinSolver_->GetMallocSize()
                  << ", real size:" << memAllocatinSolver_->GetSize();
    return AsdOps::Status::OkStatus();
}

AsdOps::Status Plan::RunNodeInTensorViewFuncs(size_t nodeId, RunnerGraphNode &node)
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
            node.runnerVariantPack.inTensors.at(i) = viewTensor;
        } else {
            node.runnerVariantPack.inTensors.at(i) = *node.inTensors.at(i);
        }
    }
    return AsdOps::Status::OkStatus();
}

void Plan::InferShapeNode(size_t nodeId, RunnerGraphNode &node)
{
    ASD_LOG(INFO) << name_ << " node[" << nodeId << "] infer shape start";
    for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); ++i) {
        ASD_LOG(INFO) << name_ << " " << node.runner->GetName() << " intensor[" << i << "] "
                      << TensorUtil::AsdOpsTensorToString(node.runnerVariantPack.inTensors.at(i));
    }
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    node.operation->InferShape(node.runnerVariantPack.inTensors, outTensorDescs);
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
        node.runnerVariantPack.outTensors.at(i) = *outTensor;
        ASD_LOG(INFO) << name_ << " " << node.runner->GetName() << " mem solve, outTensors[" << i << "] "
                      << TensorUtil::AsdOpsTensorToString(*outTensor);
    }

    auto it = maxNodeIdTensorMap_.find(nodeId);
    if (it != maxNodeIdTensorMap_.end()) {
        for (auto tensorIt : it->second) {
            ASD_LOG(INFO) << name_ << " " << node.runner->GetName() << " free tensor:" << tensorIt;
            memAllocatinSolver_->Free((char *)tensorIt->data);
        }
    }
}

AsdOps::Status Plan::SetupAllRunners()
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        AsdOps::Status st = node.runner->Setup(node.runnerVariantPack);
        if (!st.Ok()) {
            ASD_LOG(ERROR) << name_ << " node[" << nodeId << "] setup fail, error:" << st.Message();
            return st;
        }
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] setup success";
    }
    ASD_LOG(INFO) << name_ << " setup all node success";

    tilingBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerTilingBufferSize = node.runner->GetTilingBufferSize();
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] tiling buffer size:" << runnerTilingBufferSize;
        totalTilingBufferSize_ += runnerTilingBufferSize;
        tilingBufferSizes_.at(nodeId) = runnerTilingBufferSize;
    }
    ASD_LOG(INFO) << name_ << " total node tiling buffer size:" << totalTilingBufferSize_;

    totalHostTilingBuffer_.resize(totalTilingBufferSize_);
    uint64_t tilingOffset = 0;
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        void *hostTilingBuffer = totalHostTilingBuffer_.data() + tilingOffset;
        node.runner->FillHostTilingBufferSize(hostTilingBuffer, tilingBufferSizes_.at(nodeId));
        tilingOffset += tilingBufferSizes_.at(nodeId);
    }
    ASD_LOG(INFO) << name_ << " fill all node host tiling buffer";

    workspaceBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerWorkspaceBufferSize = node.runner->GetWorkspaceBufferSize();
        workspaceBufferSizes_.at(nodeId) = runnerWorkspaceBufferSize;
        maxWorkspaceBufferSize_ = std::max(maxWorkspaceBufferSize_, runnerWorkspaceBufferSize);
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] workspace buffer size:" << runnerWorkspaceBufferSize
                      << ", max:" << maxWorkspaceBufferSize_;
    }

    intermediateBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerIntermediateBufferSize = node.runner->GetIntermediateBufferSize();
        intermediateBufferSizes_.at(nodeId) = runnerIntermediateBufferSize;
        maxIntermediateBufferSize_ = std::max(maxIntermediateBufferSize_, runnerIntermediateBufferSize);
        ASD_LOG(INFO) << name_ << " node[" << nodeId << "] intermediate buffer size:" << runnerIntermediateBufferSize
                      << ", max:" << maxIntermediateBufferSize_;
    }

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer