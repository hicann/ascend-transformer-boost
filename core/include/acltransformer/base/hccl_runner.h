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
#ifndef ACLTRANSFORMER_HCCLRUNNER_H
#define ACLTRANSFORMER_HCCLRUNNER_H

#include "acltransformer/runner.h"
#include "acltransformer/runner_type.h"
#include <asdops/types.h>
#ifdef USE_HCCL_RUNNER
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include "acltransformer/share_memory.h"
#endif

namespace AclTransformer {
#ifdef USE_HCCL_RUNNER
constexpr int MAXRANKSIZE = 8;
struct CommInitInfo {
    bool barrier[MAXRANKSIZE];
    int signal = 0;
    HcclRootInfo hcclCommId;
    CommInitInfo()
    {
        for (int i = 0; i < MAXRANKSIZE; i++) {
            barrier[i] = false;
        }
    }
};
#endif

class HcclRunner : public Runner {
public:
    HcclRunner(const std::string &name, RunnerType runnerType = RUNNER_TYPE_UNDEFINED, int rank = 0, int rankSize = 0,
               int rankRoot = 0);
    virtual ~HcclRunner();

protected:
    AsdOps::Status SetupImpl(const RunnerVariantPack &runnerVariantPack) override;
    uint64_t GetTilingBufferSizeImpl() override;
    void FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize) override;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    uint64_t GetIntermediateBufferSizeImpl() override;
    AsdOps::Status ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack) override;
#ifdef USE_HCCL_RUNNER
    HcclReduceOp GetAllReduceType(std::string allReduceType);
#endif

private:
#ifdef USE_HCCL_RUNNER
    HcclDataType GetLcclDtype(AsdOps::TensorDType dtype);
    void ShmGetHcclCommId(CShareMemory &shm, const CommInitInfo &shmInfo);
    void ShmSetHcclCommId(CShareMemory &shm, CommInitInfo &shmInfo);
    void ShmBarrier(CShareMemory &shm, CommInitInfo &shmInfo);
    void ShmSetReady(CShareMemory &shm, CommInitInfo &shmInfo);
#endif

protected:
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
    int rank_ = 0;
    int rankSize_ = 0;
    int rankRoot_ = 0;
#ifdef USE_HCCL_RUNNER
    HcclReduceOp allReduceType_ = HCCL_REDUCE_SUM;
    HcclComm hcclComm_;
    HcclRootInfo hcclCommId_;
#endif
};
} // namespace AclTransformer
#endif