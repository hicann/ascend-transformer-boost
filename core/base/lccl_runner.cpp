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
#include "acltransformer/base/lccl_runner.h"
#include "acltransformer/lccl_comm_pool.h"

#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/log/log.h>

namespace AclTransformer {
LcclRunner::LcclRunner(const std::string &name, RunnerType runnerType, int rank, int rankSize) : Runner(name), runnerType_(runnerType)
{
    #ifdef USE_LCCL_RUNNER
    ASD_LOG(INFO) << "LcclRunner::LcclRunner called";
    std::map<int, Lccl::LcclComm*>& lcclCommPool = AsdOps::GetSingleton<LcclCommPool>().lcclCommPool;
    if (lcclCommPool.find(rank) == lcclCommPool.end()) {
        ASD_LOG(INFO) << "new Lccl Comm of rank begin: " << rank;
        Lccl::LcclComm* newLcclComm = new Lccl::LcclComm(rank, rankSize);
        if (newLcclComm == nullptr) {
            ASD_LOG(ERROR) << "new Lccl Comm of rank fail: " << rank;
        } else {
            lcclCommPool[rank] = newLcclComm;
            ASD_LOG(INFO) << "new Lccl Comm of rank success: " << rank;
            bool createContext = false;
            newLcclComm->LcclCommInit(createContext);
            ASD_LOG(INFO) << "init Lccl Comm of rank success: " << rank;
        }
    }
    rank_ = rank;
    #endif
}

LcclRunner::~LcclRunner() {}

AsdOps::Status LcclRunner::SetupImpl(const RunnerVariantPack &runnerVariantPack)
{
    return AsdOps::Status::OkStatus();
}

uint64_t LcclRunner::LcclRunner::GetTilingBufferSizeImpl() 
{
    return 0;
}

void LcclRunner::LcclRunner::FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize)
{
    
}

uint64_t LcclRunner::LcclRunner::GetWorkspaceBufferSizeImpl()
{
    return 0;
}

uint64_t LcclRunner::LcclRunner::GetIntermediateBufferSizeImpl()
{
    return 0;
}

uint64_t LcclRunner::GetLcclDtype(AsdOps::TensorDType dtype)
{
    #ifdef USE_LCCL_RUNNER
    switch (dtype) {
        case AsdOps::TENSOR_DTYPE_FLOAT: return Lccl::lcclFloat32;
        case AsdOps::TENSOR_DTYPE_FLOAT16: return Lccl::lcclFloat16;
        case AsdOps::TENSOR_DTYPE_INT8: return Lccl::lcclInt8;
        case AsdOps::TENSOR_DTYPE_INT32: return Lccl::lcclInt32;
        case AsdOps::TENSOR_DTYPE_UINT8: return Lccl::lcclUint8;
        default: return -1;
    }
    #endif
    return -1;
}

AsdOps::Status LcclRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    #ifdef USE_LCCL_RUNNER
    switch (runnerType_) {
        case RUNNER_TYPE_ALL_REDUCE:
            AsdOps::GetSingleton<LcclCommPool>().lcclCommPool[rank_]
                ->Allreduce(runnerVariantPack.inTensors[0].data, runnerVariantPack.outTensors[0].data, 
                runnerVariantPack.inTensors[0].Numel(), 
                static_cast<Lccl::lcclDataType_t>(GetLcclDtype(runnerVariantPack.inTensors[0].desc.dtype)),
                Lccl::lcclSum, handle.stream);
            break;
        default:
            break;
    }
    #endif
    return AsdOps::Status::OkStatus();
}
}
