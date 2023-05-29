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
#include "acltransformer/base/ops_runner.h"
#include <algorithm>
#include <sstream>
#include <json/json.h>
#include <asdops/ops.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/mem_allocation_solver/best_mem_allocation_solver.h"
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
static std::string TensorToString(const AsdOps::Tensor &tensor)
{
    std::ostringstream ss;
    ss << "dtype:" << tensor.desc.dtype << ", format:" << tensor.desc.format << ", numel:" << tensor.Numel()
       << ", dataSize:" << tensor.dataSize << ", data:" << tensor.data;
    return ss.str();
}

OpsRunner::OpsRunner(const std::string &name) : Runner(name) { memAllocatinSolver_ = new BestMemAllocationSolver(); }

OpsRunner::~OpsRunner()
{
    if (memAllocatinSolver_) {
        delete memAllocatinSolver_;
        memAllocatinSolver_ = nullptr;
    }
}

AsdOps::Status OpsRunner::Init()
{
    InitTensorMaxNodeMap();
    LogKernelGraph();
    return AsdOps::Status::OkStatus();
}

AsdOps::Status OpsRunner::Setup(Handle &handle, VariantPack &variantPack)
{
    Reset();

    int ret = Plan(handle, variantPack);
    if (ret != 0) {
        ASD_LOG(ERROR) << "Plan fail";
        return AsdOps::Status::FailStatus(1, "plan fail");
    }

    FillTilingData(variantPack);
    return AsdOps::Status::OkStatus();
}

uint64_t OpsRunner::GetWorkspaceSize() { return intermediateSize_ + tilingData_.size() + workspaceSize_; }

AsdOps::Status OpsRunner::Execute(Handle &handle, VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " execute start, intermediateSize:" << intermediateSize_
                  << ", tilingSize:" << tilingData_.size() << ", workspaceSize:" << workspaceSize_;

    char *deviceIntermediateBuffer = static_cast<char *>(variantPack.workspace);
    for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
        auto &node = kernelGraph_.nodes.at(i);
        for (uint64_t tensorId = 0; tensorId < node.runInfo.GetInTensorCount(); tensorId++) {
            AsdOps::Tensor &tensor = node.runInfo.GetInTensor(tensorId);
            if (IsInternalTensor(node.inTensors.at(tensorId))) {
                tensor.data = deviceIntermediateBuffer + (uint64_t)tensor.data;
            } else {
                int64_t tensorIdInRuninfo = GetInTensorId(node.inTensors.at(tensorId));
                tensor.data = variantPack.inTensors.at(tensorIdInRuninfo).data;
            }
        }
        for (uint64_t tensorId = 0; tensorId < node.runInfo.GetOutTensorCount(); tensorId++) {
            AsdOps::Tensor &tensor = node.runInfo.GetOutTensor(tensorId);
            if (IsInternalTensor(node.outTensors.at(tensorId))) {
                tensor.data = deviceIntermediateBuffer + (uint64_t)tensor.data;
            } else {
                int64_t tensorIdInRuninfo = GetOutTensorId(node.outTensors.at(tensorId));
                tensor.data = variantPack.outTensors.at(tensorIdInRuninfo).data;
            }
        }
    }

    uint64_t offset = intermediateSize_;
    if (tilingData_.size() > 0) {
        char *deviceTilingBuffer = static_cast<char *>(variantPack.workspace) + offset;
        int st = AsdRtMemCopy(deviceTilingBuffer, tilingData_.size(), tilingData_.data(), tilingData_.size(),
                              ASDRT_MEMCOPY_HOST_TO_DEVICE);
        if (st != ASDRT_SUCCESS) {
            return AsdOps::Status::FailStatus(1, "copy device memory fail");
        }

        uint64_t internalOffset = 0;
        for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
            AsdOps::RunInfo &kernelRunInfo = kernelGraph_.nodes.at(i).runInfo;
            uint64_t tilingBufferSize = kernelTilingSizes_.at(i);
            kernelRunInfo.SetDeviceLaunchBuffer(static_cast<char *>(deviceTilingBuffer) + internalOffset,
                                                tilingBufferSize);
            internalOffset += tilingBufferSize;
        }
        offset += tilingData_.size();
    }

    if (workspaceSize_ > 0) {
        char *deviceWorkspaceBuffer = static_cast<char *>(variantPack.workspace) + offset;
        for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
            AsdOps::RunInfo &kernelRunInfo = kernelGraph_.nodes.at(i).runInfo;
            const AsdOps::SVector<int64_t> &workspaces = kernelRunInfo.GetWorkSpace();
            AsdOps::SVector<void *> deviceLaunchBufferWorkspace(workspaces.size());
            uint64_t internalOffset = 0;
            for (size_t i = 0; workspaces.size(); ++i) {
                deviceLaunchBufferWorkspace[i] = deviceWorkspaceBuffer + internalOffset;
                internalOffset += workspaces[i];
            }
            kernelRunInfo.SetDeviceLaunchBufferWorkspace(deviceLaunchBufferWorkspace);
        }
    }

    for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
        auto &node = kernelGraph_.nodes.at(i);
        AsdOps::Kernel *kernel = node.kernel;
        AsdOps::RunInfo &kernelRunInfo = node.runInfo;
        ASD_LOG(INFO) << kernel->GetName() << " run start";
        LogRunInfo(kernelRunInfo);
        kernel->Run(kernelRunInfo);
        ASD_LOG(INFO) << kernel->GetName() << " run start";
    }
    ASD_LOG(INFO) << GetName() << " execute end";

    return AsdOps::Status::OkStatus();
}

void OpsRunner::Reset()
{
    intermediateSize_ = 0;
    tilingData_.clear();
    kernelTilingSizes_.clear();
    workspaceSize_ = 0;
    memAllocatinSolver_->Reset();
}

int OpsRunner::Plan(Handle &handle, const VariantPack &variantPack)
{
    kernelGraph_.inTensors = variantPack.inTensors;
    kernelGraph_.outTensors = variantPack.outTensors;

    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        auto &node = kernelGraph_.nodes.at(nodeId);
        const AsdOps::OpDesc &opDesc = node.opDesc;
        AsdOps::Operation *op = AsdOps::Ops::Instance().GetOperationByName(opDesc.opName);
        if (op == nullptr) {
            ASD_LOG(ERROR) << "get operation by name fail, opName:" << opDesc.opName;
            return 1;
        }

        node.runInfo.SetStream(handle.stream);
        node.runInfo.SetOpDesc(opDesc);

        for (const auto tensorIt : node.inTensors) {
            node.runInfo.AddInTensor(*tensorIt);
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            AsdOps::Tensor tensor;
            node.runInfo.AddOutTensor(tensor);
        }
        ASD_LOG(INFO) << opDesc.opName << " infer shape start";
        LogRunInfo(node.runInfo);
        AsdOps::Status st = op->InferShape(node.runInfo);
        if (!st.Ok()) {
            ASD_LOG(ERROR) << opDesc.opName << " infer shape fail, error:" << st.Message();
            return 1;
        }
        ASD_LOG(INFO) << opDesc.opName << " infer shape success";

        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            AsdOps::Tensor *outTensor = node.outTensors.at(i);
            AsdOps::Tensor &runInfoOutTensor = node.runInfo.GetOutTensor(i);
            if (IsInternalTensor(outTensor)) {
                outTensor->desc = runInfoOutTensor.desc;
                outTensor->dataSize = CalcTensorDataSize(runInfoOutTensor);
                outTensor->data = memAllocatinSolver_->Malloc(outTensor->dataSize);
            }
            runInfoOutTensor = *outTensor;
        }
        LogRunInfo(node.runInfo);

        AsdOps::Tactic *tactic = op->GetBestTactic(node.runInfo);
        if (tactic == nullptr) {
            ASD_LOG(ERROR) << opDesc.opName << " get best tactic fail";
            return 1;
        }

        node.kernel = tactic->GetBestKernel(node.runInfo);
        if (node.kernel == nullptr) {
            ASD_LOG(ERROR) << tactic->GetName() << " get best kernel fail, kernel count:" << tactic->GetKernelCount();
            return 1;
        }
        ASD_LOG(INFO) << opDesc.opName << " best tactic:" << tactic->GetName()
                      << ", best kernel:" << node.kernel->GetName();

        auto it = maxNodeIdTensorMap_.find(nodeId);
        if (it != maxNodeIdTensorMap_.end()) {
            for (auto tensorIt : it->second) {
                memAllocatinSolver_->Free((char *)tensorIt->data);
            }
        }
    }

    intermediateSize_ = memAllocatinSolver_->GetSize();
    ASD_LOG(INFO) << "MemAllocationSolver malloc size:" << memAllocatinSolver_->GetMallocSize()
                  << ", real size:" << intermediateSize_;

    return 0;
}

void OpsRunner::FillTilingData(const VariantPack &variantPack)
{
    uint64_t maxKernelWorkspaceSize = 0;
    uint64_t offset = 0;
    for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
        auto &node = kernelGraph_.nodes.at(i);
        AsdOps::Kernel *kernel = node.kernel;
        AsdOps::RunInfo &kernelRunInfo = node.runInfo;
        uint64_t tilingSize = kernel->GetLaunchBufferSize(kernelRunInfo);
        kernelTilingSizes_.push_back(tilingSize);
        if (tilingSize > 0) {
            tilingData_.resize(tilingData_.size() + tilingSize);
            kernel->InitHostLaunchBuffer(kernelRunInfo, static_cast<char *>(tilingData_.data()) + offset, tilingSize);
            offset += tilingSize;

            const AsdOps::SVector<int64_t> &workspaces = kernelRunInfo.GetWorkSpace();
            uint64_t kernelWorkspaceSize = 0;
            for (size_t j = 0; j < workspaces.size(); ++j) {
                kernelWorkspaceSize += workspaces[i];
            }
            maxKernelWorkspaceSize = std::max(maxKernelWorkspaceSize, kernelWorkspaceSize);
        }
    }
    workspaceSize_ = maxKernelWorkspaceSize;
}

void OpsRunner::InitTensorMaxNodeMap()
{
    for (size_t i = 0; i < kernelGraph_.internalTensors.size(); ++i) {
        AsdOps::Tensor &internalTensor = kernelGraph_.internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
            auto &node = kernelGraph_.nodes.at(nodeId);
            for (auto inTensorIt : node.inTensors) {
                if (&internalTensor == inTensorIt) {
                    maxNodeId = nodeId;
                    dependNodeCount++;
                }
            }
        }
        tensorMaxNodeIdMap_[&internalTensor] = maxNodeId;
        ASD_LOG(INFO) << "internal tensor[" << i << "] maxNodeId:" << maxNodeId
                      << ", dependNodeCount:" << dependNodeCount;
        ASD_LOG_IF(dependNodeCount == 0, ERROR) << "internal tensor[" << i << "] dependNodeCount is 0, graph wrong";
        maxNodeIdTensorMap_[maxNodeId].insert(&internalTensor);
    }
}

bool OpsRunner::IsInternalTensor(const AsdOps::Tensor *tensor)
{
    for (auto &internalTensor : kernelGraph_.internalTensors) {
        if (tensor == &internalTensor) {
            return true;
        }
    }

    return false;
}

int64_t OpsRunner::GetInTensorId(const AsdOps::Tensor *tensor)
{
    for (size_t i = 0; i < kernelGraph_.inTensors.size(); ++i) {
        if (&kernelGraph_.inTensors.at(i) == tensor) {
            return i;
        }
    }
    return -1;
}

int64_t OpsRunner::GetOutTensorId(const AsdOps::Tensor *tensor)
{
    for (size_t i = 0; i < kernelGraph_.outTensors.size(); ++i) {
        if (&kernelGraph_.outTensors.at(i) == tensor) {
            return i;
        }
    }
    return -1;
}

void OpsRunner::LogRunInfo(const AsdOps::RunInfo &kernelRunInfo)
{
    ASD_LOG(INFO) << "runinfo.opdesc.opName:" << kernelRunInfo.GetOpDesc().opName;
    ASD_LOG(INFO) << "runinfo.stream:" << kernelRunInfo.GetStream();
    for (size_t i = 0; i < kernelRunInfo.GetInTensorCount(); ++i) {
        ASD_LOG(INFO) << "runinfo.intensor[" << i << "]: " << TensorToString(kernelRunInfo.GetInTensor(i));
    }
    for (size_t i = 0; i < kernelRunInfo.GetOutTensorCount(); ++i) {
        ASD_LOG(INFO) << "runinfo.outtensor[" << i << "]: " << TensorToString(kernelRunInfo.GetOutTensor(i));
    }
}

void OpsRunner::LogKernelGraph()
{
    for (size_t i = 0; i < kernelGraph_.inTensors.size(); ++i) {
        ASD_LOG(INFO) << "graph inTensors[" << i << "]: " << &kernelGraph_.inTensors[i];
    }
    for (size_t i = 0; i < kernelGraph_.outTensors.size(); ++i) {
        ASD_LOG(INFO) << "graph outTensors[" << i << "]: " << &kernelGraph_.outTensors[i];
    }
    for (size_t i = 0; i < kernelGraph_.internalTensors.size(); ++i) {
        ASD_LOG(INFO) << "graph internalTensors[" << i << "]: " << &kernelGraph_.internalTensors[i];
    }
    for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
        for (size_t j = 0; j < kernelGraph_.nodes[i].inTensors.size(); ++j) {
            ASD_LOG(INFO) << "graph node[" << i << "] inTensors[" << j << "]: " << kernelGraph_.nodes[i].inTensors[j];
        }
        for (size_t j = 0; j < kernelGraph_.nodes[i].outTensors.size(); ++j) {
            ASD_LOG(INFO) << "graph node[" << i << "] outTensors[" << j << "]: " << kernelGraph_.nodes[i].outTensors[j];
        }
    }
}
} // namespace AclTransformer