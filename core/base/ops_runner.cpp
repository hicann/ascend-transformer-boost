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
#include "acltransformer/base/ops_runner.h"
#include <algorithm>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <asdops/ops.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/filesystem/filesystem.h>
#include "acltransformer/utils/mem_allocation_solver/best_mem_allocation_solver.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "acltransformer/kernel_cache.h"
#include "acltransformer/statistic.h"

namespace AclTransformer {
std::string KernelGraph::ToString() const
{
    std::stringstream ss;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]: " << &inTensors[i] << " " << TensorUtil::AsdOpsTensorToString(inTensors[i])
           << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]: " << &outTensors[i] << " " << TensorUtil::AsdOpsTensorToString(outTensors[i])
           << std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]: " << &internalTensors[i] << " "
           << TensorUtil::AsdOpsTensorToString(internalTensors[i]) << std::endl;
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = 0; j < nodes[i].inTensors.size(); ++j) {
            ss << "node[" << i << "] inTensors[" << j << "]: " << nodes[i].inTensors[j] << " "
               << TensorUtil::AsdOpsTensorToString(*nodes[i].inTensors[j]) << std::endl;
        }
        for (size_t j = 0; j < nodes[i].outTensors.size(); ++j) {
            ss << "node[" << i << "] outTensors[" << j << "]: " << nodes[i].outTensors[j] << " "
               << TensorUtil::AsdOpsTensorToString(*nodes[i].outTensors[j]) << std::endl;
        }
    }
    return ss.str();
}

std::string AsdOpsRunInfoToString(const AsdOps::RunInfo &kernelRunInfo)
{
    std::stringstream ss;
    ss << "opdesc.opName:" << kernelRunInfo.GetOpDesc().opName << ", stream:" << kernelRunInfo.GetStream() << std::endl;

    for (size_t i = 0; i < kernelRunInfo.GetInTensorCount(); ++i) {
        ss << "intensors[" << i << "]: " << TensorUtil::AsdOpsTensorToString(kernelRunInfo.GetInTensor(i)) << std::endl;
    }
    for (size_t i = 0; i < kernelRunInfo.GetOutTensorCount(); ++i) {
        ss << "outtensors[" << i << "]: " << TensorUtil::AsdOpsTensorToString(kernelRunInfo.GetOutTensor(i));
        if (i != kernelRunInfo.GetOutTensorCount() - 1) {
            ss << std::endl;
        }
    }

    return ss.str();
}

OpsRunner::OpsRunner(const std::string &name, RunnerType runnerType) : Runner(name), runnerType_(runnerType)
{
    memAllocatinSolver_ = new BestMemAllocationSolver();
}

OpsRunner::~OpsRunner()
{
    if (memAllocatinSolver_) {
        delete memAllocatinSolver_;
        memAllocatinSolver_ = nullptr;
    }
}

AsdOps::Status OpsRunner::SetupImpl(const VariantPack &variantPack)
{
    Reset();

    AsdOps::Status st = SetupKernelGraph(variantPack);
    if (!st.Ok()) {
        return st;
    }

    AsdOps::GetSingleton<KernelCache>().Init(runnerType_, kernelGraph_.nodes.size());

    InitTensorMaxNodeMap();
    ASD_LOG(INFO) << GetName() << " Setup start, kernel graph:\n" << kernelGraph_.ToString();

    if (!PlanKernelGraph(variantPack)) {
        ASD_LOG(ERROR) << GetName() << " PlanKernelGraph fail";
        return AsdOps::Status::FailStatus(1, "PlanKernelGraph fail");
    }

    FillTilingData(variantPack);
    return AsdOps::Status::OkStatus();
}

uint64_t OpsRunner::GetWorkspaceSizeImpl() { return intermediateSize_ + tilingData_.size() + workspaceSize_; }

AsdOps::Status OpsRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " execute start, intermediateSize:" << intermediateSize_
                  << ", tilingSize:" << tilingData_.size() << ", workspaceSize:" << workspaceSize_;
    UpdateRunInfoTensorData(variantPack);
    UpdateRunInfoTiling(variantPack);
    UpdateRunInfoWorkspace(variantPack);

    RunAllKernel(handle);

    ASD_LOG(INFO) << GetName() << " execute end";

    return AsdOps::Status::OkStatus();
}

void OpsRunner::UpdateRunInfoTensorData(VariantPack &variantPack)
{
    char *deviceIntermediateBuffer = static_cast<char *>(variantPack.workspace);
    for (auto &node : kernelGraph_.nodes) {
        for (uint64_t tensorId = 0; tensorId < node.kernelRunInfo.GetInTensorCount(); tensorId++) {
            AsdOps::Tensor &tensor = node.kernelRunInfo.GetInTensor(tensorId);
            if (IsInternalTensor(node.inTensors.at(tensorId))) {
                tensor.data = deviceIntermediateBuffer + (uint64_t)tensor.data;
            } else {
                int64_t tensorIdInVariantPack = GetInTensorId(node.inTensors.at(tensorId));
                if (tensorIdInVariantPack != -1) {
                    tensor.data = variantPack.inTensors.at(tensorIdInVariantPack).data;
                } else {
                    int64_t tensorIdInVariantPack = GetOutTensorId(node.inTensors.at(tensorId));
                    if (tensorIdInVariantPack != -1) {
                        tensor.data = variantPack.outTensors.at(tensorIdInVariantPack).data;
                    } else {
                        ASD_LOG(ERROR) << GetName() << " node.inTensors[" << tensorId
                                       << "] not in variantPack's inTensors or outTensors";
                    }
                }
            }
        }
        for (uint64_t tensorId = 0; tensorId < node.kernelRunInfo.GetOutTensorCount(); tensorId++) {
            AsdOps::Tensor &tensor = node.kernelRunInfo.GetOutTensor(tensorId);
            if (IsInternalTensor(node.outTensors.at(tensorId))) {
                tensor.data = deviceIntermediateBuffer + (uint64_t)tensor.data;
            } else {
                int64_t tensorIdInVariantPack = GetOutTensorId(node.outTensors.at(tensorId));
                if (tensorIdInVariantPack != -1) {
                    tensor.data = variantPack.outTensors.at(tensorIdInVariantPack).data;
                } else {
                    ASD_LOG(ERROR) << GetName() << " node.outTensors[" << tensorId
                                   << "] not in variantPack's inTensors or outTensors";
                }
            }
        }
    }
}

AsdOps::Status OpsRunner::UpdateRunInfoTiling(VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " update kernel runinfo launch buffer";
    if (tilingData_.size() > 0) {
        AsdOps::Timer timer;
        char *deviceTilingBuffer = static_cast<char *>(variantPack.workspace) + intermediateSize_;
        int st = AsdRtMemCopy(deviceTilingBuffer, tilingData_.size(), tilingData_.data(), tilingData_.size(),
                              ASDRT_MEMCOPY_HOST_TO_DEVICE);
        AsdOps::GetSingleton<Statistic>().tillingCopyTime += timer.ElapsedMicroSecond();
        if (st != ASDRT_SUCCESS) {
            ASD_LOG(ERROR) << "AsdRtMemCopy fail, error:" << st;
            return AsdOps::Status::FailStatus(1, "AsdRtMemCopy fail");
        }

        uint64_t internalOffset = 0;
        for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
            AsdOps::RunInfo &kernelRunInfo = kernelGraph_.nodes.at(i).kernelRunInfo;
            uint64_t tilingBufferSize = kernelTilingSizes_.at(i);
            kernelRunInfo.SetDeviceLaunchBuffer(static_cast<char *>(deviceTilingBuffer) + internalOffset,
                                                tilingBufferSize);
            internalOffset += tilingBufferSize;
        }
    }
    return AsdOps::Status::OkStatus();
}

void OpsRunner::UpdateRunInfoWorkspace(VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " update kernel runinfo workspace";
    if (workspaceSize_ > 0) {
        char *deviceWorkspaceBuffer =
            static_cast<char *>(variantPack.workspace) + intermediateSize_ + tilingData_.size();
        for (auto &node : kernelGraph_.nodes) {
            AsdOps::RunInfo &kernelRunInfo = node.kernelRunInfo;
            const AsdOps::SVector<int64_t> &workspaces = kernelRunInfo.GetWorkSpace();
            ASD_LOG(INFO) << GetName() << " " << node.kernel->GetName() << " workspaces.size:" << workspaces.size();
            AsdOps::SVector<void *> deviceLaunchBufferWorkspace(workspaces.size());
            uint64_t internalOffset = 0;
            for (size_t i = 0; i < workspaces.size(); ++i) {
                deviceLaunchBufferWorkspace[i] = deviceWorkspaceBuffer + internalOffset;
                ASD_LOG(INFO) << GetName() << " " << node.kernel->GetName() << " deviceLaunchBufferWorkspace[" << i
                              << "]:" << deviceLaunchBufferWorkspace[i];
                internalOffset += workspaces[i];
            }

            kernelRunInfo.SetDeviceLaunchBufferWorkspace(deviceLaunchBufferWorkspace);
        }
    }
}

void OpsRunner::RunAllKernel(Handle &handle)
{
    ASD_LOG(INFO) << GetName() << " start run all kernel, kernel count:" << kernelGraph_.nodes.size();
    for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
        auto &node = kernelGraph_.nodes.at(i);
        AsdOps::Kernel *kernel = node.kernel;
        if (kernel == nullptr) {
            ASD_LOG(ERROR) << GetName() << " node[" << i << "] kernel is null";
            return;
        }
        AsdOps::RunInfo &kernelRunInfo = node.kernelRunInfo;
        kernelRunInfo.SetStream(handle.stream);
        ASD_LOG(INFO) << GetName() << " " << kernel->GetName() << " run start, runinfo:\n"
                      << AsdOpsRunInfoToString(kernelRunInfo);

        if (AsdOps::GetSingleton<Config>().IsSkipKernel(kernel->GetName())) {
            ASD_LOG(INFO) << GetName() << " " << kernel->GetName() << " skip";
            continue;
        }
        kernel->Run(kernelRunInfo);

        if (AsdOps::GetSingleton<Config>().IsSaveTensor()) {
            ASD_LOG(INFO) << GetName() << " " << kernel->GetName()
                          << " AsdRtStreamSynchronize, stream:" << handle.stream;
            int ret = AsdRtStreamSynchronize(handle.stream);
            ASD_LOG_IF(ret != 0, ERROR) << GetName() << " " << kernel->GetName()
                                        << " AsdRtStreamSynchronize fail, ret:" << ret;
            std::string dirPath =
                Config::GetSaveTensorDir() + "/" + GetName() + "/" + std::to_string(i) + "_" + kernel->GetName();
            TensorUtil::SaveRunInfo(handle, kernelRunInfo, dirPath);
            ASD_LOG(INFO) << GetName() << " " << kernel->GetName() << " SaveRunInfo " << dirPath;
        }
        ASD_LOG(INFO) << GetName() << " " << kernel->GetName() << " run end";
    }
    ASD_LOG(INFO) << GetName() << " finish run all kernel";
}

void OpsRunner::Reset()
{
    intermediateSize_ = 0;
    tilingData_.clear();
    kernelTilingSizes_.clear();
    workspaceSize_ = 0;
    memAllocatinSolver_->Reset();
    for (auto &node : kernelGraph_.nodes) {
        node.kernelRunInfo.Reset();
    }
}

bool OpsRunner::PlanKernelGraph(const VariantPack &variantPack)
{
    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        if (!PlanOneKernel(nodeId)) {
            return false;
        }
    }

    intermediateSize_ = memAllocatinSolver_->GetSize();
    ASD_LOG(INFO) << GetName() << " MemAllocationSolver malloc size:" << memAllocatinSolver_->GetMallocSize()
                  << ", real size:" << intermediateSize_;

    return true;
}

bool OpsRunner::PlanOneKernel(size_t nodeId)
{
    auto &node = kernelGraph_.nodes.at(nodeId);
    const AsdOps::OpDesc &opDesc = node.opDesc;

    AsdOps::Operation *op = AsdOps::Ops::Instance().GetOperationByName(node.opDesc.opName);
    if (op == nullptr) {
        ASD_LOG(ERROR) << GetName() << " get operation by name fail, opName:" << node.opDesc.opName;
        return false;
    }

    if (!PlanOneKernelBuildRunInfo(node, nodeId)) {
        return false;
    }
    if (!PlanOneKernelInferShape(op, node, nodeId)) {
        return false;
    }

    if (!PlanOneKernelSelectBestKernel(op, node, nodeId)) {
        return false;
    }

    auto it = maxNodeIdTensorMap_.find(nodeId);
    if (it != maxNodeIdTensorMap_.end()) {
        for (auto tensorIt : it->second) {
            memAllocatinSolver_->Free((char *)tensorIt->data);
            ASD_LOG(INFO) << GetName() << " " << opDesc.opName << " mem free:" << tensorIt->data;
        }
    }

    return true;
}

bool OpsRunner::PlanOneKernelBuildRunInfo(KernelGraphNode &node, size_t nodeId)
{
    node.kernelRunInfo.SetOpDesc(node.opDesc);
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        AsdOps::Tensor *tensor = node.inTensors.at(i);
        if (i < node.inTensorViewFuncs.size() && node.inTensorViewFuncs.at(i)) {
            AsdOps::Tensor viewTensor = *tensor;
            viewTensor.desc.dims.clear();
            ASD_LOG(INFO) << GetName() << " node[" << nodeId << "] inTensorViewFuncs[" << i
                          << "], tensor->desc.dims:" << TensorUtil::AsdOpsDimsToString(tensor->desc.dims)
                          << ",  viewTensor.desc.dims:" << TensorUtil::AsdOpsDimsToString(viewTensor.desc.dims);
            node.inTensorViewFuncs.at(i)(tensor->desc.dims, viewTensor.desc.dims);
            if (viewTensor.Numel() != tensor->Numel()) {
                ASD_LOG(ERROR) << GetName() << " node[" << nodeId << "] inTensorViewFuncs[" << i
                               << "], viewTensor.Numel:" << viewTensor.Numel() << ", tensor.Numel:" << tensor->Numel();
                return false;
            }
            ASD_LOG(INFO) << GetName() << " node[" << nodeId << " view inTensor[" << i
                          << "], old:" << TensorUtil::AsdOpsDimsToString(tensor->desc.dims)
                          << ", new:" << TensorUtil::AsdOpsDimsToString(viewTensor.desc.dims);
            node.kernelRunInfo.AddInTensor(viewTensor);
        } else {
            node.kernelRunInfo.AddInTensor(*tensor);
        }
    }
    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        AsdOps::Tensor tensor;
        node.kernelRunInfo.AddOutTensor(tensor);
    }
    return true;
}

bool OpsRunner::PlanOneKernelInferShape(AsdOps::Operation *op, KernelGraphNode &node, size_t nodeId)
{
    ASD_LOG(INFO) << GetName() << " " << op->GetName()
                  << " infer shape start, runInfo:" << AsdOpsRunInfoToString(node.kernelRunInfo);
    if (node.inferShapePreFunc) {
        ASD_LOG(INFO) << GetName() << " " << op->GetName()
                      << " call inferShapePreFunc, old kernelRunInfo:" << AsdOpsRunInfoToString(node.kernelRunInfo);
        node.inferShapePreFunc(node.kernelRunInfo);
        ASD_LOG(INFO) << GetName() << " " << op->GetName()
                      << " call inferShapePreFunc, new kernelRunInfo:" << AsdOpsRunInfoToString(node.kernelRunInfo);
    }
    AsdOps::Status st = op->InferShape(node.kernelRunInfo);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << op->GetName() << " infer shape fail, error:" << st.Message();
        return false;
    }
    ASD_LOG(INFO) << GetName() << " " << op->GetName()
                  << " infer shape success, runInfo:" << AsdOpsRunInfoToString(node.kernelRunInfo);

    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        AsdOps::Tensor *outTensor = node.outTensors.at(i);
        AsdOps::Tensor &runInfoOutTensor = node.kernelRunInfo.GetOutTensor(i);
        if (IsInternalTensor(outTensor)) {
            if (runInfoOutTensor.desc.dims.size() != 0) {
                outTensor->desc = runInfoOutTensor.desc;
            } else {
                ASD_LOG(INFO) << GetName() << " " << op->GetName() << " outTensors[" << i
                              << "] is internal tensor, infer shape wrong, not use infer shape desc";
            }
            outTensor->dataSize = TensorUtil::CalcTensorDataSize(outTensor->desc);
            outTensor->data = memAllocatinSolver_->Malloc(TensorUtil::AlignInt(outTensor->dataSize, 32));
            ASD_LOG(INFO) << GetName() << " " << op->GetName() << " outTensors[" << i
                          << "] is internal tensor, mem solve:" << outTensor->data;
        } else {
            ASD_LOG(INFO) << GetName() << " " << op->GetName() << " outTensors[" << i << "] is not internal tensor";
        }
        runInfoOutTensor.data = outTensor->data;
        runInfoOutTensor.dataSize = outTensor->dataSize;
    }

    ASD_LOG(INFO) << GetName() << " " << op->GetName() << " after mem solve, runinfo:\n"
                  << AsdOpsRunInfoToString(node.kernelRunInfo);
    return true;
}

bool OpsRunner::PlanOneKernelSelectBestKernel(AsdOps::Operation *op, KernelGraphNode &node, size_t nodeId)
{
    if (AsdOps::GetSingleton<Config>().IsKernelCacheEnable()) {
        node.kernel = AsdOps::GetSingleton<KernelCache>().Get(runnerType_, nodeId, node.kernelRunInfo);
        if (node.kernel) {
            ASD_LOG(INFO) << GetName() << " " << op->GetName() << ", get cached best kernel:" << node.kernel->GetName();
            return true;
        }
    }

    AsdOps::Tactic *tactic = op->GetBestTactic(node.kernelRunInfo);
    if (tactic == nullptr) {
        ASD_LOG(ERROR) << GetName() << " " << op->GetName()
                       << " get best tactic fail, tactic count:" << op->GetTacticCount();
        return false;
    }
    ASD_LOG(INFO) << GetName() << " best tactic:" << tactic->GetName();

    AsdOps::Timer timer;
    node.kernel = tactic->GetBestKernel(node.kernelRunInfo);
    AsdOps::GetSingleton<Statistic>().getBestKernelTime += timer.ElapsedMicroSecond();
    if (node.kernel == nullptr) {
        ASD_LOG(ERROR) << GetName() << " " << tactic->GetName()
                       << " get best kernel fail, kernel count:" << tactic->GetKernelCount();
        return false;
    }
    ASD_LOG(INFO) << GetName() << " " << op->GetName() << " get best tactic:" << tactic->GetName()
                  << ", best kernel:" << node.kernel->GetName();

    if (AsdOps::GetSingleton<Config>().IsKernelCacheEnable()) {
        AsdOps::GetSingleton<KernelCache>().Add(runnerType_, nodeId, node.kernelRunInfo, node.kernel);
        ASD_LOG(INFO) << GetName() << " " << op->GetName() << ", cache best kernel:" << node.kernel->GetName();
    }
    return true;
}

void OpsRunner::FillTilingData(const VariantPack &variantPack)
{
    kernelTilingSizes_.resize(kernelGraph_.nodes.size());
    uint64_t totalTilingSize = 0;
    for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
        KernelGraphNode &node = kernelGraph_.nodes.at(i);
        AsdOps::Kernel *kernel = node.kernel;
        AsdOps::RunInfo &kernelRunInfo = node.kernelRunInfo;
        uint64_t orgTilingSize = kernel->GetLaunchBufferSize(kernelRunInfo);
        uint64_t tilingSize = TensorUtil::AlignInt(orgTilingSize, 32);
        ;
        ASD_LOG(INFO) << GetName() << " " << kernel->GetName() << " orgTilingSize:" << orgTilingSize
                      << ", tilingSize:" << tilingSize;
        kernelTilingSizes_.at(i) = tilingSize;
        totalTilingSize += tilingSize;
    }

    tilingData_.resize(totalTilingSize);
    uint64_t offset = 0;
    for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
        KernelGraphNode &node = kernelGraph_.nodes.at(i);
        AsdOps::Kernel *kernel = node.kernel;
        AsdOps::RunInfo &kernelRunInfo = node.kernelRunInfo;
        uint64_t tilingSize = kernelTilingSizes_.at(i);
        if (tilingSize > 0) {
            kernel->InitHostLaunchBuffer(kernelRunInfo, static_cast<char *>(tilingData_.data()) + offset, tilingSize);
            if (AsdOps::GetSingleton<Config>().IsSaveTensor()) {
                std::string fileDir =
                    Config::GetSaveTensorDir() + "/" + GetName() + "/" + std::to_string(i) + "_" + kernel->GetName();
                WriteTilingData(static_cast<char *>(tilingData_.data()) + offset, tilingSize, fileDir);
            }
            offset += tilingSize;
        }
    }

    uint64_t maxKernelWorkspaceSize = 0;
    for (size_t i = 0; i < kernelGraph_.nodes.size(); ++i) {
        KernelGraphNode &node = kernelGraph_.nodes.at(i);
        AsdOps::Kernel *kernel = node.kernel;
        AsdOps::RunInfo &kernelRunInfo = node.kernelRunInfo;
        uint64_t tilingSize = kernelTilingSizes_.at(i);
        if (tilingSize > 0) {
            const AsdOps::SVector<int64_t> &workspaces = kernelRunInfo.GetWorkSpace();
            ASD_LOG(INFO) << GetName() << " " << kernel->GetName() << " workspaces.size:" << workspaces.size();
            uint64_t kernelWorkspaceSize = 0;
            for (size_t i = 0; i < workspaces.size(); ++i) {
                kernelWorkspaceSize += workspaces.at(i);
                ASD_LOG(INFO) << GetName() << " " << kernel->GetName() << " workspaces[" << i
                              << "]:" << workspaces.at(i);
            }
            kernelWorkspaceSize = TensorUtil::AlignInt(int64_t(kernelWorkspaceSize), 32);
            ASD_LOG(INFO) << GetName() << " " << kernel->GetName() << ", kernelWorkspaceSize:" << kernelWorkspaceSize
                          << ", maxKernelWorkspaceSize:" << maxKernelWorkspaceSize;
            ASD_LOG_IF(kernelWorkspaceSize > 1024 * 1024, ERROR)
                << GetName() << " " << kernel->GetName() << " kernelWorkspaceSize too large, discard";
            maxKernelWorkspaceSize = std::max(maxKernelWorkspaceSize, kernelWorkspaceSize);
        }
    }
    workspaceSize_ = maxKernelWorkspaceSize;
}

void OpsRunner::WriteTilingData(const char *tilingData, size_t len, const std::string &dirPath)
{
    AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    std::string filePath = dirPath + "/" + "tilingdata.bin";

    std::ofstream fd(filePath.c_str(), std::ios::binary);
    if (!fd.is_open()) {
        ASD_LOG(ERROR) << "write tiling file fail, path:" << filePath;
        return;
    }
    fd.write(tilingData, len);
    fd.close();
    ASD_LOG(INFO) << "write tiling success";
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
        ASD_LOG(INFO) << GetName() << " internal tensor[" << i << "] maxNodeId:" << maxNodeId
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
} // namespace AclTransformer