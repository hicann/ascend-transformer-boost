/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)

#include "op_def.h"
#include "allgather.h"
#include "91093/allgather_hierarchy_double_ring.h"
#include "allreduce_one_shot.h"
#include "allreduce_two_shot.h"
#include "allreduce_big_data.h"
#include "91093/allreduce_big_data_sio.h"
#include "91093/allreduce_hierarchy_double_ring.h"
#include "91093/reduce_scatter_big_data_91093_4step.h"
#include "91093/reduce_scatter_hierarchy_double_ring.h"
#include "91093/all2all_hierarchy.h"
#include "91093/all2all_hierarchy_small.h"

#include "../kernels/lcal_allreduce_2npu_read.cce"
#include "../kernels/lcal_allreduce_2npu_write.cce"
#include "../kernels/lcal_allreduce_2npu_big_write.cce"
#include "../kernels/lcal_allreduce_two_shot.cce"
#include "../kernels/lcal_allreduce_big_data.cce"
#include "../kernels/lcal_allreduce_two_shot_910B2C.cce"
#include "../kernels/lcal_allreduce_big_data_910B2C.cce"
#include "../kernels/lcal_allreduce_deterministic.cce"
#include "../kernels/lcal_allreduce_deterministic_big_data.cce"
#include "../kernels/lcal_reduce_scatter_big_data_write.cce"
#include "../kernels/lcal_reduce_scatter_write.cce"
#include "../kernels/lcal_reduce_scatter.cce"
#include "../kernels/lcal_reduce_scatter_big_data.cce"
#include "../kernels/lcal_allgather_910B2C.cce"
#include "../kernels/lcal_allgather_big_data_910B2C.cce"
#include "../kernels/lcal_allgather_2npu.cce"
#include "../kernels/lcal_allgather_2npu_big_data_write.cce"
#include "../kernels/lcal_allgather.cce"
#include "../kernels/lcal_allgather_big_data.cce"
#include "../kernels/lcal_broadcast_write.cce"
#include "../kernels/lcal_broadcast_big_data.cce"
#include "../kernels/lcal_all2all_transpose.cce"

#define CLASS_OP_910B_RDMA_LAUNCH(name, type) \
do { \
name<type> opKernel(localRank, localRankSize, extraFlag); \
opKernel.Init(KERNELS_ARGS_CALL()); \
opKernel.Process();                 \
} while (0)

extern "C" __global__ __aicore__ __attribute__((section("Attr_Section_Lcal"))) void LcalDescriptor() {}

#define LCCL_BROADCAST_FUNC_AUTO_DEF(suffix) \
extern "C" __global__ __aicore__ void LcalBroadcast##suffix(KERNEL_ARGS_FUN()) |
{ \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    __gm__ char * shareAddrs[LCAL_MAX_RANK_SIZE]; \
    GET_IPC_MEM_ARGS(char); \
    if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) { \
        LcalBroadcast2npuBigDataWrite(ALLREDUCE_ARGS_CALL(char));   \
    } else { \
        LcalBroadcastBigData(ALLREDUCE_ARGS_CALL(char));   \
    } \
    } \
}


