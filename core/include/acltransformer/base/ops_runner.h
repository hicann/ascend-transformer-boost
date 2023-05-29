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
#ifndef ACLTRANSFORMER_OPSRUNNER_H
#define ACLTRANSFORMER_OPSRUNNER_H
#include "acltransformer/runner.h"
#include <vector>
#include <map>
#include <set>
#include <asdops/kernel.h>
#include <asdops/op_desc.h>
#include <asdops/run_info.h>

namespace AclTransformer {
struct KernelGraphNode {
    AsdOps::OpDesc opDesc;
    std::vector<AsdOps::Tensor *> inTensors;
    std::vector<AsdOps::Tensor *> outTensors;
    AsdOps::Kernel *kernel = nullptr;
    AsdOps::RunInfo runInfo;
};

struct KernelGraph {
    std::vector<AsdOps::Tensor> inTensors;
    std::vector<AsdOps::Tensor> outTensors;
    std::vector<AsdOps::Tensor> internalTensors;
    std::vector<KernelGraphNode> nodes;
};

class MemAllocationSolver;

class OpsRunner : public Runner {
public:
    OpsRunner(const std::string &name);
    virtual ~OpsRunner();
    AsdOps::Status Init() override;
    AsdOps::Status Setup(Handle &handle, VariantPack &runInfo) override;
    uint64_t GetWorkspaceSize() override;
    AsdOps::Status Execute(Handle &handle, VariantPack &variantPack) override;

private:
    void Reset();
    int Plan(Handle &handle, const VariantPack &variantPack);
    void FillTilingData(const VariantPack &variantPack);
    void InitTensorMaxNodeMap();
    bool IsInternalTensor(const AsdOps::Tensor *tensor);
    int64_t GetInTensorId(const AsdOps::Tensor *tensor);
    int64_t GetOutTensorId(const AsdOps::Tensor *tensor);
    void LogRunInfo(const AsdOps::RunInfo &kernelRunInfo);
    void LogKernelGraph();

protected:
    KernelGraph kernelGraph_;
    uint64_t intermediateSize_ = 0;
    std::vector<char> tilingData_;
    std::vector<uint64_t> kernelTilingSizes_;
    uint64_t workspaceSize_ = 0;

    std::map<AsdOps::Tensor *, uint64_t> tensorMaxNodeIdMap_;
    std::map<uint64_t, std::set<AsdOps::Tensor *>> maxNodeIdTensorMap_;
    MemAllocationSolver *memAllocatinSolver_ = nullptr;
};
} // namespace AclTransformer
#endif