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
#ifndef ACLTRANSFORMER_OPSRUNNER_H
#define ACLTRANSFORMER_OPSRUNNER_H
#include "acltransformer/runner.h"
#include <vector>
#include <functional>
#include <map>
#include <set>
#include <asdops/kernel.h>
#include <asdops/op_desc.h>
#include <asdops/run_info.h>
#include <asdops/operation.h>
#include "acltransformer/runner_type.h"

namespace AclTransformer {
using ViewFunc = std::function<void(const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)>;
using InferShapePreFunc = std::function<void(AsdOps::RunInfo &runInfo)>;
using KernelGrapModifyFunc = std::function<void(const RunnerVariantPack &runnerVariantPack)>;

enum TensorType {
    UNDEFINED_TENSOR = 0,
    IN_TENSOR,
    OUT_TENSOR,
    INTERMEDIATE_TENSOR,
};

struct KernelGraphNode {
    AsdOps::OpDesc opDesc;
    AsdOps::SVector<AsdOps::Tensor *> inTensors;
    AsdOps::SVector<AsdOps::Tensor *> outTensors;
    AsdOps::SVector<ViewFunc> inTensorViewFuncs;
    AsdOps::Kernel *kernel = nullptr;
    AsdOps::RunInfo kernelRunInfo;
    InferShapePreFunc inferShapePreFunc;
    AsdOps::SVector<TensorType> inTensorsType;
    AsdOps::SVector<TensorType> outTensorsType;
};

struct KernelGraph {
    AsdOps::SVector<AsdOps::Tensor> inTensors;
    AsdOps::SVector<AsdOps::Tensor> outTensors;
    AsdOps::SVector<AsdOps::Tensor, 64> internalTensors;
    AsdOps::SVector<KernelGraphNode, 64> nodes;
    KernelGrapModifyFunc kernelGraphModifyFunc;
    std::string ToString() const;
};

class MemAllocationSolver;

class OpsRunner : public Runner {
public:
    OpsRunner(const std::string &name, RunnerType runnerType = RUNNER_TYPE_UNDEFINED);
    virtual ~OpsRunner();

protected:
    virtual AsdOps::Status SetupKernelGraph(const RunnerVariantPack &runnerVariantPack);

protected:
    AsdOps::Status SetupImpl(const RunnerVariantPack &runnerVariantPack) override;
    uint64_t GetTilingBufferSizeImpl() override;
    void FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize) override;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    uint64_t GetIntermediateBufferSizeImpl() override;
    AsdOps::Status ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack) override;

private:
    void Reset();
    bool PlanKernelGraph(const RunnerVariantPack &runnerVariantPack);
    bool PlanOneKernel(size_t nodeId);
    bool PlanOneKernelBuildRunInfo(KernelGraphNode &node, size_t nodeId);
    bool PlanOneKernelInferShape(AsdOps::Operation *op, KernelGraphNode &node, size_t nodeId);
    bool PlanOneKernelSelectBestKernel(AsdOps::Operation *op, KernelGraphNode &node, size_t nodeId);
    void CalcTilingBufferSize(const RunnerVariantPack &runnerVariantPack);
    void InitTensorMaxNodeMap();
    bool IsInternalTensor(const AsdOps::Tensor *tensor);
    void WriteTilingData(const char *tilingData, size_t len, const std::string &filePath);
    void UpdateRunInfoTensorData(RunnerVariantPack &runnerVariantPack);
    AsdOps::Status UpdateRunInfoTiling(RunnerVariantPack &runnerVariantPack);
    void UpdateRunInfoWorkspace(RunnerVariantPack &runnerVariantPack);
    void RunAllKernel(Handle &handle);
    bool IsRunnerVariantPackInputEqual(const RunnerVariantPack &runnerVariantPack1,
                                       const RunnerVariantPack &runnerVariantPack2);
    void InitTensorsType();
    void CalcKernelWorkspace();
#ifdef USE_PROFILING
    void ReportLaunchInfo(const uint64_t beginTime, const char *opName, size_t nodeId);
    void ReportAdditionalInfo(const uint64_t timeStamp, const char *opName, size_t nodeId);
#endif

protected:
    KernelGraph kernelGraph_;
    uint64_t totalTilingSize_ = 0;
    AsdOps::SVector<uint64_t, 64> tilingSizes_;
    uint64_t workspaceSize_ = 0;
    uint64_t intermediateSize_ = 0;
    std::map<AsdOps::Tensor *, uint64_t> tensorMaxNodeIdMap_;
    std::map<uint64_t, std::set<AsdOps::Tensor *>> maxNodeIdTensorMap_;
    MemAllocationSolver *memAllocatinSolver_ = nullptr;
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
    RunnerVariantPack lastRunnerVariantPack_;
    AsdOps::RunInfo lastRunInfo_;
    AsdOps::Kernel *lastKernel_ = nullptr;
    bool initTensorTypeFlag_ = false;
    bool setupCacheEnable_ = true;
};
} // namespace AclTransformer
#endif