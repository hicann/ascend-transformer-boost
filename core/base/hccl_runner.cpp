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
#include "acltransformer/base/hccl_runner.h"
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/statistic.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
HcclRunner::HcclRunner(const std::string &name, RunnerType runnerType, int rank, int rankSize, int rankRoot)
    : Runner(name), runnerType_(runnerType), rank_(rank), rankSize_(rankSize), rankRoot_(rankRoot)
{
#ifdef USE_HCCL_RUNNER
    if (rankSize <= MAXRANKSIZE) {
        ASD_LOG(INFO) << "HCCL Runner Init Begin , rank : " << rank;
        CShareMemory shm("hcclShareMem", sizeof(AclTransformer::CommInitInfo));
        auto *shmInfo = (AclTransformer::CommInitInfo *)shm.GetShm();
        ASD_LOG(INFO) << "create share memory success , rank : " << rank;
        if (rank == rankRoot) {
            auto ret = HcclGetRootInfo(&hcclCommId_);
            if (ret != HCCL_SUCCESS) {
                ASD_LOG(ERROR) << "HCCL GetRootInfo ERROR" << ret;
            }
            ShmSetHcclCommId(shm, *shmInfo);
        } else {
            ShmGetHcclCommId(shm, *shmInfo);
        }

        ASD_LOG(INFO) << "Access share memory success , rank : " << rank;
        ShmBarrier(shm, *shmInfo);
        ASD_LOG(INFO) << "Barrier success , rank : " << rank;
        auto ret = HcclCommInitRootInfo(rankSize, &hcclCommId_, rank, &hcclComm_);
        if (ret != HCCL_SUCCESS && ret != HCCL_E_PARA) {
            ASD_LOG(ERROR) << "HCCL CommInitRootInfo ERROR" << ret;
        }
        ASD_LOG(INFO) << "HcclRunner init success , rank : " << rank;
    } else {
        ASD_LOG(ERROR) << "rankSize too big: " << rankSize;
    }
#endif
}

HcclRunner::~HcclRunner()
{
#ifdef USE_HCCL_RUNNER
    auto ret = HcclCommDestroy(hcclComm_);
    if (ret != HCCL_SUCCESS) {
        ASD_LOG(ERROR) << "HCCL CommSestroy ERROR" << ret;
    }
#endif
}

AsdOps::Status HcclRunner::SetupImpl(const RunnerVariantPack &runnerVariantPack) { return AsdOps::Status::OkStatus(); }

uint64_t HcclRunner::HcclRunner::GetTilingBufferSizeImpl() { return 0; }

void HcclRunner::HcclRunner::FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize) {}

uint64_t HcclRunner::HcclRunner::GetWorkspaceBufferSizeImpl() { return 0; }

uint64_t HcclRunner::HcclRunner::GetIntermediateBufferSizeImpl() { return 0; }

#ifdef USE_HCCL_RUNNER
HcclDataType HcclRunner::GetLcclDtype(AsdOps::TensorDType dtype)
{
    switch (dtype) {
    case AsdOps::TENSOR_DTYPE_FLOAT: return HCCL_DATA_TYPE_FP32;
    case AsdOps::TENSOR_DTYPE_FLOAT16: return HCCL_DATA_TYPE_FP16;
    case AsdOps::TENSOR_DTYPE_INT8: return HCCL_DATA_TYPE_INT8;
    case AsdOps::TENSOR_DTYPE_INT32: return HCCL_DATA_TYPE_INT32;
    default: return HCCL_DATA_TYPE_RESERVED;
    }
}

void HcclRunner::ShmGetHcclCommId(CShareMemory &shm, const CommInitInfo &shmInfo)
{
    bool commIdReady = false;
    while (true) {
        shm.SemLock();
        if (shmInfo.signal != 0) {
            hcclCommId_ = shmInfo.hcclCommId;
            commIdReady = true;
        }
        shm.SemUnLock();
        if (commIdReady) {
            break;
        }
    }
}

void HcclRunner::ShmSetHcclCommId(CShareMemory &shm, CommInitInfo &shmInfo)
{
    shm.SemLock();
    shmInfo.hcclCommId = hcclCommId_;
    shmInfo.signal = 1;
    shm.SemUnLock();
}

void HcclRunner::ShmSetReady(CShareMemory &shm, CommInitInfo &shmInfo)
{
    shm.SemLock();
    shmInfo.barrier[rank_] = true;
    shm.SemUnLock();
}

void HcclRunner::ShmBarrier(CShareMemory &shm, CommInitInfo &shmInfo)
{
    ShmSetReady(shm, shmInfo);
    bool allReady = true;
    while (true) {
        shm.SemLock();
        for (int i = 0; i <= rankSize_; i++) {
            if (!shmInfo.barrier[i]) {
                allReady = false;
            }
        }
        shm.SemUnLock();
        if (allReady) {
            break;
        }
    }
}
#endif

AsdOps::Status HcclRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
#ifdef USE_HCCL_RUNNER
    AsdOps::Timer timer;
    if (runnerType_ == RUNNER_TYPE_ALL_REDUCE) {
        auto ret = HcclAllReduce(runnerVariantPack.inTensors[0].data, runnerVariantPack.outTensors[0].data,
                                 runnerVariantPack.inTensors[0].Numel(),
                                 GetLcclDtype(runnerVariantPack.inTensors[0].desc.dtype), HCCL_REDUCE_SUM, hcclComm_,
                                 handle.stream);
        if (ret != HCCL_SUCCESS) {
            ASD_LOG(ERROR) << "HcclAllReduce ERROR" << ret;
        }
    }
#endif
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
