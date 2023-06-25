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

struct KernelGraphNode {
    AsdOps::OpDesc opDesc;
    AsdOps::SVector<AsdOps::Tensor *> inTensors;
    AsdOps::SVector<AsdOps::Tensor *> outTensors;
    AsdOps::SVector<ViewFunc> inTensorViewFuncs;
    AsdOps::Kernel *kernel = nullptr;
    AsdOps::RunInfo kernelRunInfo;
    InferShapePreFunc inferShapePreFunc;
};

struct KernelGraph {
    AsdOps::SVector<AsdOps::Tensor> inTensors;
    AsdOps::SVector<AsdOps::Tensor> outTensors;
    AsdOps::SVector<AsdOps::Tensor, 64> internalTensors;
    AsdOps::SVector<KernelGraphNode, 64> nodes;
    std::string ToString() const;
};

class MemAllocationSolver;

class OpsRunner : public Runner {
public:
    OpsRunner(const std::string &name, RunnerType runnerType = RUNNER_TYPE_UNDEFINED);
    virtual ~OpsRunner();

protected:
    virtual AsdOps::Status SetupKernelGraph(const VariantPack &variantPack);

protected:
    AsdOps::Status SetupImpl(const VariantPack &variantPack) override;

    uint64_t GetWorkspaceSizeImpl() override;
    AsdOps::Status ExecuteImpl(Handle &handle, VariantPack &variantPack) override;

private:
    void Reset();
    bool PlanKernelGraph(const VariantPack &variantPack);
    bool PlanOneKernel(size_t nodeId);
    bool PlanOneKernelBuildRunInfo(KernelGraphNode &node, size_t nodeId);
    bool PlanOneKernelInferShape(AsdOps::Operation *op, KernelGraphNode &node, size_t nodeId);
    bool PlanOneKernelSelectBestKernel(AsdOps::Operation *op, KernelGraphNode &node, size_t nodeId);
    void FillTilingData(const VariantPack &variantPack);
    void InitTensorMaxNodeMap();
    bool IsInternalTensor(const AsdOps::Tensor *tensor);
    int64_t GetInTensorId(const AsdOps::Tensor *tensor);
    int64_t GetOutTensorId(const AsdOps::Tensor *tensor);
    void WriteTilingData(const char *tilingData, size_t len, const std::string &filePath);
    void UpdateRunInfoTensorData(VariantPack &variantPack);
    AsdOps::Status UpdateRunInfoTiling(VariantPack &variantPack);
    void UpdateRunInfoWorkspace(VariantPack &variantPack);
    void RunAllKernel(Handle &handle);

protected:
    KernelGraph kernelGraph_;
    uint64_t intermediateSize_ = 0;
    std::vector<char> tilingData_;
    std::vector<uint64_t> kernelTilingSizes_;
    uint64_t workspaceSize_ = 0;
    std::map<AsdOps::Tensor *, uint64_t> tensorMaxNodeIdMap_;
    std::map<uint64_t, std::set<AsdOps::Tensor *>> maxNodeIdTensorMap_;
    MemAllocationSolver *memAllocatinSolver_ = nullptr;
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
};
} // namespace AclTransformer
#endif