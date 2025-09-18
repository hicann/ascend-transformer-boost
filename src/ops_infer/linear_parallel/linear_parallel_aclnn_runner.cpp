/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "linear_parallel_aclnn_runner.h"
#include "atb/utils/dl_manager.h"
#include "atb/utils/aclnn_util.h"
#include <hccl/hccl.h>
// #include <aclnnop/aclnn_matmul_reduce_scatter_v2.h>
// #include <aclnnop/aclnn_all_gather_matmul_v2.h>
// #include <aclnnop/aclnn_all_to_all_all_gather_batch_matmul.h>
// #include <aclnnop/aclnn_grouped_mat_mul_allto_allv_v2.h>

namespace atb {

static const uint32_t LINEAR_REDUCE_SCATTER_IN_TENSOR_NUM = 6;
static const uint32_t LINEAR_REDUCE_SCATTER_OUT_TENSOR_NUM = 2;

// static const uint32_t ALL_GATHER_LINEAR_IN_TENSOR_NUM = 6;
// static const uint32_t ALL_GATHER_LINEAR_OUT_TENSOR_NUM = 3;

// static const uint32_t ALLTOALLVC_ALL_GATHER_GMM_IN_TENSOR_NUM = 8;
// static const uint32_t ALLTOALLVC_ALL_GATHER_GMM_OUT_TENSOR_NUM = 4;

// static const uint32_t GMM_REDUCE_SCATTER_ALLTOALLVC_IN_TENSOR_NUM = 9;
// static const uint32_t GMM_REDUCE_SCATTER_ALLTOALLVC_OUT_TENSOR_NUM = 2;

static const uint32_t BIAS_TENSOR_INDEX = 2;
static const uint32_t GATHER_OUT_TENSOR_INDEX = 1;

aclnnStatus (*LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, const char *, const char *, int64_t, int64_t, int64_t, const char *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **) = nullptr;

aclnnStatus (*LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2Func_)(void *, uint64_t, aclOpExecutor *,
                                                                          aclrtStream) = nullptr;

// aclnnStatus (*aclnnAllGatherMatmulV2GetWorkspaceSizeFunc_)(
//     const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
//     int64_t, const char *, int64_t, int64_t, int64_t, int64_t, int64_t, const char *, const aclTensor *,
//     const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **) = nullptr;

// aclnnStatus (*aclnnAllGatherMatmulV2Func_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

// aclnnStatus (*aclnnAlltoAllvGroupedMatmulV2GetWorkspaceSizeFunc_)(
//     const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
//     const aclTensor *, const aclTensor *, const char *, int64_t, const aclIntArray *, const aclIntArray *, bool,
//     bool, bool, int64_t, int64_t, const char *, const aclTensor *, const aclTensor *, const aclTensor *, const
//     aclTensor *, uint64_t *, aclOpExecutor **) = nullptr;

// aclnnStatus (*aclnnAlltoAllvGroupedMatmulV2Func_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

// aclnnStatus (*aclnnGroupedMatmulAlltoAllvV2GetWorkspaceSizeFunc_)(
//     const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
//     const aclTensor *, const aclTensor *, const aclTensor *, const char *, const char *, int64_t, int64_t, int64_t,
//     const aclIntArray *, const aclIntArray *, bool, bool, const aclTensor *, const aclTensor *, uint64_t *,
//     aclOpExecutor **) = nullptr;

// aclnnStatus (*aclnnGroupedMatmulAlltoAllvV2Func_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

LinearParallelAclnnRunner::LinearParallelAclnnRunner(const infer::LinearParallelParam &param, bool useRankTableFile)
    : AclnnRunner("LinearParallelAclnnRunner", RUNNER_TYPE_LINEAR_PARALLEL),
      hcclRunner_(!useRankTableFile ? HcclRunner("LinearParallelAclnnRunner", RUNNER_TYPE_LINEAR_PARALLEL, param.rank,
                                                 param.rankSize, param.rankRoot, param.commDomain) :
                                      HcclRunner("LinearParallelAclnnRunner", RUNNER_TYPE_LINEAR_PARALLEL, param.rank,
                                                 param.rankTableFile, param.commDomain)),
      param_(param)
{
    ATB_LOG(INFO) << "LinearParallelAclnnRunner::LinearParallelAclnnRunner called";
}

LinearParallelAclnnRunner::LinearParallelAclnnRunner(const infer::LinearParallelParam &param, HcclComm hcclComm)
    : AclnnRunner("LinearParallelAclnnRunner", RUNNER_TYPE_LINEAR_PARALLEL),
      hcclRunner_("LinearParallelAclnnRunner", hcclComm, RUNNER_TYPE_LINEAR_PARALLEL), param_(param)
{
    ATB_LOG(INFO) << "LinearParallelAclnnRunner::LinearParallelAclnnRunner ext called";
}

LinearParallelAclnnRunner::~LinearParallelAclnnRunner() {}

Status LinearParallelAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack called";
    (void)runnerVariantPack;
    Status ret = NO_ERROR;
    switch (param_.type) {
        case infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            ret = BuildAclnnVariantPackMatmulReduceScatter(runnerVariantPack);
            break;

            // case infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            //     ret = BuildAclnnVariantPackAllGatherMatmul(runnerVariantPack);
            //     break;

            // case infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM:
            //     ret = BuildAclnnVariantPackAlltoAllAllGatherBatchMatMul(runnerVariantPack);
            //     break;

            // case infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC:
            //     ret = BuildAclnnVariantPackBatchMatMulReduceScatterAlltoAll(runnerVariantPack);
            //     break;

        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported type: " << param_.type;
            return ERROR_INVALID_PARAM;
    }
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "SetAclNNWorkspaceExecutor error: " << ret;
        return ret;
    }
    return ret;
}

Status LinearParallelAclnnRunner::BuildAclnnVariantPackMatmulReduceScatter(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPackMatmulReduceScatter called";
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(LINEAR_REDUCE_SCATTER_IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(LINEAR_REDUCE_SCATTER_IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        if (i >= 2) {
            this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
            continue;
        }
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = (i != 1 || !param_.transWeight) ? GetCopyTensorStride(atbTensor.desc.shape) :
                                                                    GetTransposeTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = i;
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    this->aclnnVariantPack_.aclOutTensors.reserve(LINEAR_REDUCE_SCATTER_OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(LINEAR_REDUCE_SCATTER_OUT_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        if (i >= 1) {
            this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
            continue;
        }
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = i;
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    return ret;
}

// Status LinearParallelAclnnRunner::BuildAclnnVariantPackAllGatherMatmul(const RunnerVariantPack &runnerVariantPack)
// {
//     ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPackAllGatherMatmul called";
//     this->atbVariantPack_ = runnerVariantPack;
//     Status ret = NO_ERROR;
//     this->aclnnVariantPack_.aclInTensors.reserve(ALL_GATHER_LINEAR_IN_TENSOR_NUM);
//     this->aclnnVariantPack_.aclInTensors.resize(ALL_GATHER_LINEAR_IN_TENSOR_NUM);
//     for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
//         std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
//         if (i >= 3 || (i == BIAS_TENSOR_INDEX && !param_.hasResidual)) {
//             this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
//             continue;
//         }
//         atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
//         aclnnTensorPtr->atbTensor = atbTensor;
//         aclnnTensorPtr->strides = (i != 1 || !param_.transWeight) ? GetCopyTensorStride(atbTensor.desc.shape) :
//                                                                     GetTransposeTensorStride(atbTensor.desc.shape);
//         ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
//         if (ret != NO_ERROR) {
//             ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
//             return ret;
//         }
//         aclnnTensorPtr->tensorIdx = i;
//         aclnnTensorPtr->needUpdateTensorDataPtr = true;
//         this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
//     }

//     this->aclnnVariantPack_.aclOutTensors.reserve(ALL_GATHER_LINEAR_OUT_TENSOR_NUM);
//     this->aclnnVariantPack_.aclOutTensors.resize(ALL_GATHER_LINEAR_OUT_TENSOR_NUM);
//     for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
//         std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
//         if (i >= 2 || (i == GATHER_OUT_TENSOR_INDEX && !param_.keepIntermediate)) {
//             this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
//             continue;
//         }
//         atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
//         aclnnTensorPtr->atbTensor = atbTensor;
//         aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
//         ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
//         if (ret != NO_ERROR) {
//             ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
//             return ret;
//         }
//         aclnnTensorPtr->tensorIdx = i;
//         aclnnTensorPtr->needUpdateTensorDataPtr = true;
//         this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
//     }
//     return ret;
// }


// Status
// LinearParallelAclnnRunner::BuildAclnnVariantPackAlltoAllAllGatherBatchMatMul(const RunnerVariantPack
// &runnerVariantPack)
// {
//     ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPackAlltoAllAllGatherBatchMatMul called";
//     this->atbVariantPack_ = runnerVariantPack;
//     Status ret = NO_ERROR;
//     this->aclnnVariantPack_.aclInTensors.reserve(ALLTOALLVC_ALL_GATHER_GMM_IN_TENSOR_NUM);
//     this->aclnnVariantPack_.aclInTensors.resize(ALLTOALLVC_ALL_GATHER_GMM_IN_TENSOR_NUM);
//     for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
//         std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
//         if (i >= 2) {
//             this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
//             continue;
//         }
//         atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
//         aclnnTensorPtr->atbTensor = atbTensor;
//         aclnnTensorPtr->strides = (i != 1 || !param_.transWeight) ? GetCopyTensorStride(atbTensor.desc.shape) :
//                                                                     GetTransposeTensorStride(atbTensor.desc.shape);
//         ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
//         if (ret != NO_ERROR) {
//             ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
//             return ret;
//         }
//         aclnnTensorPtr->tensorIdx = i;
//         aclnnTensorPtr->needUpdateTensorDataPtr = true;
//         this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
//     }

//     this->aclnnVariantPack_.aclOutTensors.reserve(ALLTOALLVC_ALL_GATHER_GMM_OUT_TENSOR_NUM);
//     this->aclnnVariantPack_.aclOutTensors.resize(ALLTOALLVC_ALL_GATHER_GMM_OUT_TENSOR_NUM);
//     for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
//         std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
//         if (i >= 1) {
//             this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
//             continue;
//         }
//         atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
//         aclnnTensorPtr->atbTensor = atbTensor;
//         aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
//         ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
//         if (ret != NO_ERROR) {
//             ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
//             return ret;
//         }
//         aclnnTensorPtr->tensorIdx = i;
//         aclnnTensorPtr->needUpdateTensorDataPtr = true;
//         this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
//     }
//     return ret;
// }

// Status LinearParallelAclnnRunner::BuildAclnnVariantPackBatchMatMulReduceScatterAlltoAll(
//     const RunnerVariantPack &runnerVariantPack)
// {
//     ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPackBatchMatMulReduceScatterAlltoAll called";
//     this->atbVariantPack_ = runnerVariantPack;
//     Status ret = NO_ERROR;
//     this->aclnnVariantPack_.aclInTensors.reserve(GMM_REDUCE_SCATTER_ALLTOALLVC_IN_TENSOR_NUM);
//     this->aclnnVariantPack_.aclInTensors.resize(GMM_REDUCE_SCATTER_ALLTOALLVC_IN_TENSOR_NUM);
//     for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
//         std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
//         if (i >= 2) {
//             this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
//             continue;
//         }
//         atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
//         aclnnTensorPtr->atbTensor = atbTensor;
//         aclnnTensorPtr->strides = (i != 1 || !param_.transWeight) ? GetCopyTensorStride(atbTensor.desc.shape) :
//                                                                     GetTransposeTensorStride(atbTensor.desc.shape);
//         ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
//         if (ret != NO_ERROR) {
//             ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
//             return ret;
//         }
//         aclnnTensorPtr->tensorIdx = i;
//         aclnnTensorPtr->needUpdateTensorDataPtr = true;
//         this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
//     }

//     this->aclnnVariantPack_.aclOutTensors.reserve(GMM_REDUCE_SCATTER_ALLTOALLVC_OUT_TENSOR_NUM);
//     this->aclnnVariantPack_.aclOutTensors.resize(GMM_REDUCE_SCATTER_ALLTOALLVC_OUT_TENSOR_NUM);
//     for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
//         std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
//         if (i >= 1) {
//             this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
//             continue;
//         }
//         atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
//         aclnnTensorPtr->atbTensor = atbTensor;
//         aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
//         ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
//         if (ret != NO_ERROR) {
//             ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
//             return ret;
//         }
//         aclnnTensorPtr->tensorIdx = i;
//         aclnnTensorPtr->needUpdateTensorDataPtr = true;
//         this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
//     }
//     return ret;
// }

aclnnStatus LinearParallelAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SetAclNNWorkspaceExecutor called";
    aclnnStatus ret = 0;
    switch (param_.type) {
        case infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            ret = SetAclNNWorkspaceExecutorMatmulReduceScatter();
            break;

            // case infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            //     ret = SetAclNNWorkspaceExecutorAllGatherMatmul();
            //     break;

            // case infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM:
            //     ret = SetAclNNWorkspaceExecutorAlltoAllAllGatherBatchMatMul();
            //     break;

            // case infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC:
            //     ret = SetAclNNWorkspaceExecutorBatchMatMulReduceScatterAlltoAll();
            //     break;

        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported type: " << param_.type;
            return ERROR_INVALID_PARAM;
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "SetAclNNWorkspaceExecutor error: " << ret;
        return ret;
    }
    return ret;
}

aclnnStatus LinearParallelAclnnRunner::SetAclNNWorkspaceExecutorMatmulReduceScatter()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SetAclNNWorkspaceExecutorMatmulReduceScatter called";
    Status status = LinearParallelAclnnRunner::LoadMethodMatmulReduceScatter();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "load getWorkspace function from aclnn failed! Consider upgrade CANN first";
        return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    }
    ATB_LOG(INFO) << GetLogPrefix() << "aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorStart = 0;
    aclTensor *x1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *x2 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *bias = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *x1Scale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *x2Scale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantScale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;

    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    aclTensor *amaxOutOptional = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;

    int64_t blockSize = 0;
    char group[128] = {0};
    aclnnStatus ret = HcclGetCommName(hcclRunner_.GetHcclCommSharedPtr().get(), group);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "HcclGetCommName error: " << ret;
        return ret;
    }
    char reduceOp[128] = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    char commMode[128] = "aiv";

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << raw_executor_ptr
                  << ", then take the address of the raw ptr: " << &raw_executor_ptr;

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    ret = LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_(
        x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group, reduceOp, commTurn, streamMode, groupSize,
        commMode, output, amaxOutOptional, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);

    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

// aclnnStatus LinearParallelAclnnRunner::SetAclNNWorkspaceExecutorAllGatherMatmul()
// {
//     ATB_LOG(INFO) << GetLogPrefix() << "SetAclNNWorkspaceExecutorAllGatherMatmul called";
//     ATB_LOG(INFO) << GetLogPrefix() << "aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
//                   << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
//     size_t inTensorStart = 0;
//     aclTensor *x1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *x2 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *bias = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *x1Scale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *x2Scale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *quantScale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;

//     size_t outTensorStart = 0;
//     aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
//     aclTensor *gatherOut = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
//     aclTensor *amaxOutOptional = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;

//     int64_t blockSize = 0;
//     char group[128] = {0};
//     aclnnStatus ret = HcclGetCommName(hcclRunner_.GetHcclCommSharedPtr().get(), group);
//     if (ret != ACL_SUCCESS) {
//         ATB_LOG(ERROR) << GetLogPrefix() << "HcclGetCommName error: " << ret;
//         return ret;
//     }
//     int64_t gatherIndex = 1;
//     int64_t commTurn = 0;
//     int64_t streamMode = 1;
//     int64_t groupSize = 0;
//     char commMode[128] = "aiv";

//     aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
//     ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
//                   << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
//                   << ", raw ptr from it: " << raw_executor_ptr
//                   << ", then take the address of the raw ptr: " << &raw_executor_ptr;

//     ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

//     ret = LinearParallelAclnnRunner::aclnnAllGatherMatmulV2GetWorkspaceSizeFunc_(
//         x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group, gatherIndex, commTurn, streamMode, groupSize,
//         commMode, output, gatherOut, amaxOutOptional, &(this->atbVariantPack_.workspaceBufferSize),
//         &raw_executor_ptr);

//     bool repeatable = this->executorRepeatable_;
//     this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [repeatable](aclOpExecutor *ptr) {
//         if (ptr && repeatable) { // 可复用时才手动销毁aclOpExecutor
//             aclDestroyAclOpExecutor(ptr);
//         }
//     });
//     return ret;
// }

// aclnnStatus LinearParallelAclnnRunner::SetAclNNWorkspaceExecutorAlltoAllAllGatherBatchMatMul()
// {
//     ATB_LOG(INFO) << GetLogPrefix() << "SetAclNNWorkspaceExecutorAlltoAllAllGatherBatchMatMul called";
//     ATB_LOG(INFO) << GetLogPrefix() << "aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
//                   << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
//     size_t inTensorStart = 0;
//     aclTensor *gmmX = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *gmmWeight = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *sendCountsTensorOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *recvCountsTesnorOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *mmXOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *mmWeightOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *gmmXScaleOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *gmmWeightScaleOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;

//     size_t outTensorStart = 0;
//     aclTensor *gmmY = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
//     aclTensor *mmYOptional = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
//     aclTensor *permuteOutOptional = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
//     aclTensor *globalTokenPerExpert = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;

//     char group[128] = {0};
//     aclnnStatus ret = HcclGetCommName(hcclRunner_.GetHcclCommSharedPtr().get(), group);
//     if (ret != ACL_SUCCESS) {
//         ATB_LOG(ERROR) << GetLogPrefix() << "HcclGetCommName error: " << ret;
//         return ret;
//     }
//     int64_t epWorldSize = 0;
//     std::vector<int64_t> sendCountsList(param_.rankSize * e, A / (param_.rankSize * e));
//     std::vector<int64_t> recvCountsList(param_.rankSize * e, A / (param_.rankSize * e));
//     aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
//     aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
//     bool transGmmWeight = false;
//     bool transMmWeight = false;
//     bool permuteOutFlag = false;
//     int64_t yDtype = 0;
//     int64_t rankID = param_.rank;
//     char commMode[128] = "aiv";

//     aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
//     ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
//                   << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
//                   << ", raw ptr from it: " << raw_executor_ptr
//                   << ", then take the address of the raw ptr: " << &raw_executor_ptr;

//     ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

//     ret = LinearParallelAclnnRunner::aclnnAlltoAllvGroupedMatmulV2GetWorkspaceSizeFunc_(
//         x1, x2, globalTokenPerExprt, scale, perTokenScale, sendCountsTensorOptional, recvCountsTesnorOptional,
//         mmXOptional, mmWeightOptional, group, commMode, worldsize, rankID, M, sendCounts, recvCounts, transGmmWeight,
//         transMmWeight, output, gatherOut, amaxOutOptional, &(this->atbVariantPack_.workspaceBufferSize),
//         &raw_executor_ptr);

//     bool repeatable = this->executorRepeatable_;
//     this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [repeatable](aclOpExecutor *ptr) {
//         if (ptr && repeatable) { // 可复用时才手动销毁aclOpExecutor
//             aclDestroyAclOpExecutor(ptr);
//         }
//     });
//     return ret;
// }

// aclnnStatus LinearParallelAclnnRunner::SetAclNNWorkspaceExecutorBatchMatMulReduceScatterAlltoAll()
// {
//     ATB_LOG(INFO) << GetLogPrefix() << "SetAclNNWorkspaceExecutorBatchMatMulReduceScatterAlltoAll called";
//     ATB_LOG(INFO) << GetLogPrefix() << "aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
//                   << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
//     size_t inTensorStart = 0;
//     aclTensor *x1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *x2 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *globalTokenPerExprt = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *scale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *perTokenScale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *sendCountsTensorOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *recvCountsTesnorOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *mmXOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
//     aclTensor *mmWeightOptional = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;

//     size_t outTensorStart = 0;
//     aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
//     aclTensor *mmYOptional = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;

//     char group[128] = {0};
//     aclnnStatus ret = HcclGetCommName(hcclRunner_.GetHcclCommSharedPtr().get(), group);
//     if (ret != ACL_SUCCESS) {
//         ATB_LOG(ERROR) << GetLogPrefix() << "HcclGetCommName error: " << ret;
//         return ret;
//     }
//     char commMode[128] = "aiv";
//     int64_t worldsize = param_.rankSize;
//     int64_t rankID = param_.rank;
//     int64_t M = 0;
//     std::vector<int64_t> sendCountsList(param_.rankSize * e, A / (param_.rankSize * e));
//     std::vector<int64_t> recvCountsList(param_.rankSize * e, A / (param_.rankSize * e));
//     aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
//     aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
//     bool transGmmWeight = false;
//     bool transMmWeight = false;

//     aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
//     ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
//                   << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
//                   << ", raw ptr from it: " << raw_executor_ptr
//                   << ", then take the address of the raw ptr: " << &raw_executor_ptr;

//     ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

//     ret = LinearParallelAclnnRunner::aclnnGroupedMatmulAlltoAllvV2GetWorkspaceSizeFunc_(
//         x1, x2, globalTokenPerExprt, scale, perTokenScale, sendCountsTensorOptional, recvCountsTesnorOptional,
//         mmXOptional, mmWeightOptional, group, commMode, worldsize, rankID, M, sendCounts, recvCounts, transGmmWeight,
//         transMmWeight, output, gatherOut, amaxOutOptional, &(this->atbVariantPack_.workspaceBufferSize),
//         &raw_executor_ptr);

//     bool repeatable = this->executorRepeatable_;
//     this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [repeatable](aclOpExecutor *ptr) {
//         if (ptr && repeatable) { // 可复用时才手动销毁aclOpExecutor
//             aclDestroyAclOpExecutor(ptr);
//         }
//     });
//     return ret;
// }

Status LinearParallelAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel called";
    Status status = NO_ERROR;
    aclnnStatus ret = 0;
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    switch (param_.type) {
        case infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            status = LinearParallelAclnnRunner::LoadMethodMatmulReduceScatter();
            if (status != NO_ERROR) {
                ATB_LOG(ERROR) << GetLogPrefix()
                               << "load getWorkspace function from aclnn failed! Consider upgrade CANN first";
                return status;
            }
            ATB_LOG(INFO) << GetLogPrefix() << "aclnnMatmulReduceScatterV2 execute start.";
            ret = LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2Func_(this->atbVariantPack_.workspaceBuffer,
                                                                             this->atbVariantPack_.workspaceBufferSize,
                                                                             this->aclnnExecutor_.get(), executeStream);
            break;

            // case infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            //     ATB_LOG(INFO) << GetLogPrefix() << "aclnnAllGatherMatmulV2 execute start.";
            //     ret = LinearParallelAclnnRunner::aclnnAllGatherMatmulV2Func_(this->atbVariantPack_.workspaceBuffer,
            //                                                                  this->atbVariantPack_.workspaceBufferSize,
            //                                                                  this->aclnnExecutor_.get(),
            //                                                                  executeStream);
            //     break;

            // case infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM:
            //     ATB_LOG(INFO) << GetLogPrefix() << "aclnnAlltoallvGmm execute start.";
            //     ret = LinearParallelAclnnRunner::aclnnAlltoAllvGroupedMatmulV2Func_(
            //         this->atbVariantPack_.workspaceBuffer, this->atbVariantPack_.workspaceBufferSize,
            //         this->aclnnExecutor_.get(), executeStream);
            //     break;

            // case infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC:
            //     ATB_LOG(INFO) << GetLogPrefix() << "aclnnGroupedMatmulAlltoAllvV2 execute start.";
            //     ret = LinearParallelAclnnRunner::aclnnGroupedMatmulAlltoAllvV2Func_(
            //         this->atbVariantPack_.workspaceBuffer, this->atbVariantPack_.workspaceBufferSize,
            //         this->aclnnExecutor_.get(), executeStream);
            //     break;

        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported type: " << param_.type;
            return ERROR_INVALID_PARAM;
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn error: " << ret;
        return ERROR_CANN_ERROR;
    }
    return NO_ERROR;
}

Status LinearParallelAclnnRunner::LoadMethodMatmulReduceScatter()
{
    ATB_LOG(INFO) << "LinearParallelAclnnRunner LoadMethod";
    if (LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_ != nullptr &&
        LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2Func_ != nullptr) {
        return NO_ERROR;
    }
    static DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
    Status ret =
        dlManager.getSymbol("aclnnMatmulReduceScatterV2GetWorkspaceSize",
                            (void **)&LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnMatmulReduceScatterV2GetWorkspaceSize failed! Consider upgrade the CANN first!";
        return ret;
    }
    ret = dlManager.getSymbol("aclnnMatmulReduceScatterV2",
                              (void **)&LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2Func_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnMatmulReduceScatterV2 failed! Consider upgrade the CANN first!";
        return ret;
    }
    ATB_LOG(INFO) << "load aclnnMatmulReduceScatterV2 two-staged method success!";
    return NO_ERROR;
}

// Status LinearParallelAclnnRunner::LoadMethodAllGatherMatmul()
// {
//     ATB_LOG(INFO) << "LinearParallelAclnnRunner LoadMethod";
//     if (LinearParallelAclnnRunner::aclnnAllGatherMatmulV2GetWorkspaceSizeFunc_ != nullptr &&
//         LinearParallelAclnnRunner::aclnnAllGatherMatmulV2Func_ != nullptr) {
//         return NO_ERROR;
//     }
//     DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
//     Status ret = dlManager.getSymbol("aclnnAllGatherMatmulV2GetWorkspaceSize",
//                                      (void
//                                      **)&LinearParallelAclnnRunner::aclnnAllGatherMatmulV2GetWorkspaceSizeFunc_);
//     if (ret != NO_ERROR) {
//         ATB_LOG(ERROR) << "load aclnnAllGatherMatmulV2GetWorkspaceSize failed! Consider upgrade the CANN first!";
//         return ret;
//     }
//     ret =
//         dlManager.getSymbol("aclnnAllGatherMatmulV2", (void
//         **)&LinearParallelAclnnRunner::aclnnAllGatherMatmulV2Func_);
//     if (ret != NO_ERROR) {
//         ATB_LOG(ERROR) << "load aclnnAllGatherMatmulV2 failed! Consider upgrade the CANN first!";
//         return ret;
//     }
//     ATB_LOG(INFO) << "load aclnnAllGatherMatmulV2 two-staged method success!";
//     return NO_ERROR;
// }

// Status LinearParallelAclnnRunner::LoadMethodAlltoAllAllGatherBatchMatMul()
// {
//     ATB_LOG(INFO) << "LinearParallelAclnnRunner LoadMethod";
//     if (LinearParallelAclnnRunner::AlltoAllvGroupedMatmulV2GetWorkspaceSizeFunc_ != nullptr &&
//         LinearParallelAclnnRunner::AlltoAllvGroupedMatmulV2Func_ != nullptr) {
//         return NO_ERROR;
//     }
//     DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
//     Status ret =
//         dlManager.getSymbol("AlltoAllvGroupedMatmulV2GetWorkspaceSize",
//                             (void **)&LinearParallelAclnnRunner::AlltoAllvGroupedMatmulV2GetWorkspaceSizeFunc_);
//     if (ret != NO_ERROR) {
//         ATB_LOG(ERROR) << "load AlltoAllvGroupedMatmulV2GetWorkspaceSize failed! Consider upgrade the CANN first!";
//         return ret;
//     }
//     ret = dlManager.getSymbol("AlltoAllvGroupedMatmulV2",
//                               (void **)&LinearParallelAclnnRunner::AlltoAllvGroupedMatmulV2Func_);
//     if (ret != NO_ERROR) {
//         ATB_LOG(ERROR) << "load AlltoAllvGroupedMatmulV2 failed! Consider upgrade the CANN first!";
//         return ret;
//     }
//     ATB_LOG(INFO) << "load AlltoAllvGroupedMatmulV2 two-staged method success!";
//     return NO_ERROR;
// }

// Status LinearParallelAclnnRunner::LoadMethodBatchMatMulReduceScatterAlltoAll()
// {
//     ATB_LOG(INFO) << "LinearParallelAclnnRunner LoadMethod";
//     if (LinearParallelAclnnRunner::aclnnGroupedMatmulAlltoAllvV2GetWorkspaceSizeFunc_ != nullptr &&
//         LinearParallelAclnnRunner::aclnnGroupedMatmulAlltoAllvV2Func_ != nullptr) {
//         return NO_ERROR;
//     }
//     DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
//     Status ret =
//         dlManager.getSymbol("aclnnGroupedMatmulAlltoAllvV2GetWorkspaceSize",
//                             (void **)&LinearParallelAclnnRunner::aclnnGroupedMatmulAlltoAllvV2GetWorkspaceSizeFunc_);
//     if (ret != NO_ERROR) {
//         ATB_LOG(ERROR) << "load aclnnGroupedMatmulAlltoAllvV2GetWorkspaceSize failed! Consider upgrade the CANN first!";
//         return ret;
//     }
//     ret = dlManager.getSymbol("aclnnGroupedMatmulAlltoAllvV2",
//                               (void **)&LinearParallelAclnnRunner::aclnnGroupedMatmulAlltoAllvV2Func_);
//     if (ret != NO_ERROR) {
//         ATB_LOG(ERROR) << "load aclnnGroupedMatmulAlltoAllvV2 failed! Consider upgrade the CANN first!";
//         return ret;
//     }
//     ATB_LOG(INFO) << "load aclnnGroupedMatmulAlltoAllvV2 two-staged method success!";
//     return NO_ERROR;
// }

} // namespace atb
