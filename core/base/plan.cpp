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

Plan::Plan() { memAllocatinSolver_ = new BestMemAllocationSolver(); }

Plan::~Plan()
{
    if (memAllocatinSolver_) {
        delete memAllocatinSolver_;
        memAllocatinSolver_ = nullptr;
    }
    for(auto &node : runnerGraph_.nodes) {
        delete node.runner;
        node.runner = nullptr;
    }
}

AsdOps::Status Plan::Setup(Handle handle, const VariantPack &variantPack)
{
    ASD_LOG(INFO) << "Plan::Setup start";
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.variantPack.inTensors.resize(node.inTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            if (i < node.inTensorViewFuncs.size() && node.inTensorViewFuncs.at(i)) {
                AsdOps::Tensor viewTensor = *node.inTensors.at(i);
                viewTensor.desc.dims.clear();
                node.inTensorViewFuncs.at(i)(node.inTensors.at(i)->desc.dims, viewTensor.desc.dims);
                if (viewTensor.Numel() != node.inTensors.at(i)->Numel()) {
                    ASD_LOG(ERROR) << "Plan node[" << nodeId
                                   << "] invalid view func, viewTensor.Numel:" << viewTensor.Numel()
                                   << ", tensor.Numel:" << node.inTensors.at(i)->Numel();
                    return AsdOps::Status::FailStatus(1, "invalid view");
                }
                ASD_LOG(INFO) << "Plan node[" << nodeId << " view inTensor[" << i
                              << "], old:" << TensorUtil::AsdOpsDimsToString(node.inTensors.at(i)->desc.dims)
                              << ", new:" << TensorUtil::AsdOpsDimsToString(viewTensor.desc.dims);
                node.variantPack.inTensors.at(i) = viewTensor;
            } else {
                node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
            }
        }

        node.variantPack.outTensors.resize(node.outTensors.size());

        ASD_LOG(INFO) << runnerGraph_.name << " " << node.runner->GetName() << " infer shape start";
        for (size_t i = 0; i < node.variantPack.inTensors.size(); ++i) {
            ASD_LOG(INFO) << runnerGraph_.name << " " << node.runner->GetName() << " intensor[" << i << "] "
                          << TensorUtil::AsdOpsTensorToString(node.variantPack.inTensors.at(i));
        }
        AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
        node.operation->InferShape(node.variantPack.inTensors, outTensorDescs);
        for (size_t i = 0; i < outTensorDescs.size(); ++i) {
            ASD_LOG(INFO) << runnerGraph_.name << " " << node.runner->GetName() << " outTensorDescs[" << i << "] "
                          << TensorUtil::AsdOpsTensorDescToString(outTensorDescs.at(i));
        }
        ASD_LOG(INFO) << runnerGraph_.name << " " << node.runner->GetName() << " infer shape end";

        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            AsdOps::Tensor *outTensor = node.outTensors.at(i);
            if (outTensor->data == nullptr) {
                outTensor->desc = outTensorDescs.at(i);
                outTensor->dataSize = TensorUtil::CalcTensorDataSize(*outTensor);
                outTensor->data = memAllocatinSolver_->Malloc(TensorUtil::AlignInt(outTensor->dataSize, 32));
                ASD_LOG(INFO) << runnerGraph_.name << " " << node.runner->GetName()
                              << " MemAllocationSolver Malloc dataSize:" << outTensor->dataSize
                              << ", blockAddress:" << int64_t(outTensor->data);
            }
            node.variantPack.outTensors.at(i) = *outTensor;
            ASD_LOG(INFO) << runnerGraph_.name << " " << node.runner->GetName() << " mem solve, outTensors[" << i
                          << "] " << TensorUtil::AsdOpsTensorToString(*outTensor);
        }

        auto it = maxNodeIdTensorMap_.find(nodeId);
        if (it != maxNodeIdTensorMap_.end()) {
            for (auto tensorIt : it->second) {
                ASD_LOG(INFO) << runnerGraph_.name << node.runner->GetName() << " free tensor:" << tensorIt;
                memAllocatinSolver_->Free((char *)tensorIt->data);
            }
        }
    }
    intermediateSize_ = memAllocatinSolver_->GetSize();
    ASD_LOG(INFO) << "Plan MemAllocationSolver malloc size:" << memAllocatinSolver_->GetMallocSize()
                  << ", real size:" << memAllocatinSolver_->GetSize();

    for (auto &node : runnerGraph_.nodes) {
        ASD_LOG(INFO) << "Plan call " << node.runner->GetName() << " setup ";
        AsdOps::Status st = node.runner->Setup(node.variantPack);
        if (!st.Ok()) {
            return st;
        }
        uint64_t runnerWorkspaceSize = node.runner->GetWorkspaceSize();
        runnerWorkspaceSize = TensorUtil::AlignInt(runnerWorkspaceSize, 32);
        ASD_LOG(INFO) << "Plan get " << node.runner->GetName() << " workspace size:" << runnerWorkspaceSize;
        workspaceSize_ = std::max(runnerWorkspaceSize, workspaceSize_);
    }

    ASD_LOG(INFO) << "Plan::Setup end, intermediateSize:" << intermediateSize_ << ", workspaceSize:" << workspaceSize_;
    return AsdOps::Status::OkStatus();
}

uint64_t Plan::GetWorkspaceSize() { return workspaceSize_ + intermediateSize_; }

AsdOps::Status Plan::Execute(Handle handle, VariantPack &variantPack)
{
    if (handle.stream == nullptr) {
        ASD_LOG(ERROR) << "Plan::Execute fail, handle.stream is null";
        return AsdOps::Status::FailStatus(1, "handle stream is null");
    }
    ASD_LOG(INFO) << "Plan::Execute start, runnerGraph_.nodes:" << runnerGraph_.nodes.size();
    uint64_t offset = 0;
    if (workspaceSize_ > 0) {
        for (auto &node : runnerGraph_.nodes) {
            VariantPack &runnerVariantPack = node.variantPack;
            runnerVariantPack.workspace = variantPack.workspace;
            runnerVariantPack.workspaceSize = node.runner->GetWorkspaceSize();
        }
        offset += workspaceSize_;
    }

    char *intermediateBuffer = static_cast<char *>(variantPack.workspace) + offset;
    ASD_LOG(INFO) << "Plan update tensor.data start";
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ASD_LOG(INFO) << "Plan update tensor.data node[" << nodeId << "]";
        VariantPack &runnerVariantPack = node.variantPack;
        for (size_t i = 0; i < runnerVariantPack.inTensors.size(); ++i) {
            auto &tensor = runnerVariantPack.inTensors.at(i);
            if (IsInternalTensor(node.inTensors.at(i))) {
                tensor.data = intermediateBuffer + (uint64_t)tensor.data;
                offset += tensor.dataSize;
                ASD_LOG(INFO) << "Plan update node[" << nodeId << "].intensors[" << i
                              << "] is internal, tensor.data:" << tensor.data;
            } else {
                int64_t tensorIdInRuninfo = GetInTensorId(node.inTensors.at(i));
                tensor.data = variantPack.inTensors.at(tensorIdInRuninfo).data;
                ASD_LOG(INFO) << "Plan update node[" << nodeId << "].intensor is not internal";
            }
        }
        for (size_t i = 0; i < runnerVariantPack.outTensors.size(); ++i) {
            auto &tensor = runnerVariantPack.outTensors.at(i);
            if (IsInternalTensor(node.outTensors.at(i))) {
                tensor.data = intermediateBuffer + (uint64_t)tensor.data;
                offset += tensor.dataSize;
                ASD_LOG(INFO) << "Plan update node[" << nodeId << "].outtensor[" << i
                              << "] is internal, tensor.data:" << tensor.data;
            } else {
                int64_t tensorIdInRuninfo = GetOutTensorId(node.outTensors.at(i));
                tensor.data = variantPack.outTensors.at(tensorIdInRuninfo).data;
                ASD_LOG(INFO) << "Plan update node[" << nodeId << "].outtensor[" << i << "] is not internal";
            }
        }
    }
    ASD_LOG(INFO) << "Plan update tensor.data end";

    size_t nodeId = 0;
    for (auto &node : runnerGraph_.nodes) {
        ASD_LOG(INFO) << runnerGraph_.name << " " << node.runner->GetName() << " execute start:" << handle.stream;
        LogVariantPack(node.variantPack);
        node.runner->Execute(handle, node.variantPack);
        if (AsdOps::GetSingleton<Config>().IsStreamSyncEveryRunnerEnable()) {
            AsdOps::Timer timer;
            int ret = AsdRtStreamSynchronize(handle.stream);
            AsdOps::GetSingleton<Statistic>().tillingCopyTime += timer.ElapsedMicroSecond();
            ASD_LOG_IF(ret != 0, ERROR) << "Plan AsdRtStreamSynchronize node[" << nodeId << "] fail, ret:" << ret;
        }
        if (AsdOps::GetSingleton<Config>().IsSaveTensor()) {
            AsdRtStreamSynchronize(handle.stream);
            std::string dirPath = Config::GetSaveTensorDir() + "/" + runnerGraph_.name + "/" + std::to_string(nodeId) +
                                  "_" + node.runner->GetName();
            TensorUtil::SaveVariantPack(handle, node.variantPack, dirPath);
            ASD_LOG(INFO) << "Plan SaveVariantPack " << dirPath;
        }
        nodeId++;
    }

    if (AsdOps::GetSingleton<Config>().IsStreamSyncEveryPlanEnable()) {
        int ret = AsdRtStreamSynchronize(handle.stream);
        ASD_LOG_IF(ret != 0, ERROR) << "Plan AsdRtStreamSynchronize node[" << nodeId << "] fail, ret:" << ret;
    }

    ASD_LOG(INFO) << "Plan execute success";
    return AsdOps::Status::OkStatus();
}

void Plan::Reset()
{
    workspaceSize_ = 0;
    intermediateSize_ = 0;
    if (memAllocatinSolver_) {
        memAllocatinSolver_->Reset();
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
        ASD_LOG(INFO) << runnerGraph_.name << " internal tensor[" << i << "] maxNodeId:" << maxNodeId
                      << ", dependNodeCount:" << dependNodeCount;
        ASD_LOG_IF(dependNodeCount == 0, ERROR) << "internal tensor[" << i << "] dependNodeCount is 0, graph wrong";
        maxNodeIdTensorMap_[maxNodeId].insert(&internalTensor);
    }
}

void Plan::LogVariantPack(const VariantPack &variantPack)
{
    for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
        ASD_LOG(INFO) << "variantPack.inTensors[" << i << "] "
                      << TensorUtil::AsdOpsTensorToString(variantPack.inTensors[i]);
    }
    for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
        ASD_LOG(INFO) << "variantPack.outTensors[" << i << "] "
                      << TensorUtil::AsdOpsTensorToString(variantPack.outTensors[i]);
    }
}

} // namespace AclTransformer