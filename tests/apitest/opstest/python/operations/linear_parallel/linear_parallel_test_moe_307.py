#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import os
import json
import unittest
import sys
import torch
import torch_npu
import torch.multiprocessing as mp
from linear_parallel_moe_common import QuantGranularity, QuantInfo, CommType, CoCDataTypeDesc, MoeTestDate

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)

torch.manual_seed(0)


def main_worker(rank, comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
                data_type, quant_info, EP, TP, quant_type, out_data_ype):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')

    acl_matmul_allreduce_operation = torch.classes.OperationTorch.OperationTorch(
        "LinearParallelOperation")

    acl_param = json.dumps({"type": 5, "rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "transWeight": False, "backend": "lcoc",
                            "quantType": quant_type, "outDataType": out_data_ype,
                            "moeInfo": {"epSize": world_size, "localExpertNums":
                                local_expert_nums, "tpSize": 1}})

    acl_matmul_allreduce_operation.set_param(acl_param)
    torch.manual_seed(0)
    moedata = MoeTestDate(rank, CommType(comm_type), world_size, batch, M, K, N, trans_b, local_expert_nums,
                          CoCDataTypeDesc(data_type), quant_info, EP, TP)

    in_tensors = []
    input_tensor = moedata.matrix_a
    input_tensor = input_tensor.reshape(M, K)
    in_tensors.append(input_tensor.to(torch.device('npu')))

    weight_tensor = moedata.matrix_b
    weight_tensor = weight_tensor.reshape(local_expert_nums, K, N)
    in_tensors.append(weight_tensor.to(torch.device('npu')))
    if quant_type == 3:
        dequantScale = moedata.matrix_dequant_scale
        dequantScale = dequantScale.reshape(N * local_expert_nums)
        in_tensors.append(dequantScale.to(torch.device('npu')))

        quantScale = moedata.matrix_quant_scale
        quantScale = quantScale.reshape(M)
        in_tensors.append(quantScale.to(torch.device('npu')))

    global_tokens_per_expert_matrix = moedata.global_tokens_per_expert_matrix
    in_tensors.append(global_tokens_per_expert_matrix.to(torch.device('npu')))

    maxOutputSize = torch.zeros(input_tensor.shape[0] * world_size * local_expert_nums, dtype=torch.int32)
    in_tensors.append(maxOutputSize.to(torch.device('npu')))

    out_tensor = acl_matmul_allreduce_operation.execute(in_tensors)

    torch.npu.synchronize()
    golden_out_tensor = moedata.matrix_c
    out_tensor_compare = out_tensor[0].to(torch.device('cpu'))[:golden_out_tensor.shape[1], :]

    assert check_precision_new(out_tensor_compare, golden_out_tensor, rank)


def check_precision_new(out_tensor, golden_out_tensor, rank, err=2 ** -3):
    # 计算每个元素的误差阈值
    max_err = err * torch.max(torch.ones_like(golden_out_tensor), torch.abs(golden_out_tensor))

    # 计算实际误差
    error = torch.abs(out_tensor - golden_out_tensor)

    # 计算不满足条件的元素个数
    num_failures = torch.sum(error > max_err).item()
    if rank == 0:
        print("num_failures: ", num_failures)
    return num_failures == 0


class LinearParallelCoverOperationTest(operation_test.OperationTest):

    def test_linear_paraller_fp16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            return
        print(f"———————— LinearParallelCoverOp test start ————————")
        print("------------ALLTOALLVC ALLGATHER MATMUL Non quantitative scenarios-----------")
        world_size = 8
        comm_type = 309
        batch = 1
        M = 1024
        K = 1024
        N = 1024
        trans_b = 0
        quant_granularity = -1
        quant_group_size = -1
        has_quant_offset = -1
        dequant_group_size = -1
        local_expert_nums = 4
        EP = 8
        TP = 1
        out_data_type = 1
        dequant_granularity = -1
        has_dequant_offset = -1
        data_type = 0
        quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
                               QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)
        mp.spawn(main_worker, nprocs=world_size,
                 args=(comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
                       CoCDataTypeDesc(data_type), quant_info, EP, TP, dequant_granularity, out_data_type))

    def test_linear_paraller_fp16_quant(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            return
        print(f"———————— LinearParallelCoverOp test start ————————")
        print("------------ALLTOALLVC ALLGATHER MATMUL Quantify scenarios-----------")
        world_size = 8
        comm_type = 309
        batch = 1
        M = 1024
        K = 1024
        N = 1024
        trans_b = 0
        quant_granularity = -1
        quant_group_size = -1
        has_quant_offset = -1
        dequant_group_size = -1
        local_expert_nums = 4
        EP = 8
        TP = 1
        out_data_type = 1
        dequant_granularity = 3
        has_dequant_offset = 0
        data_type = 2
        quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
                               QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)
        mp.spawn(main_worker, nprocs=world_size,
                 args=(comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
                       CoCDataTypeDesc(data_type), quant_info, EP, TP, dequant_granularity, out_data_type))

if __name__ == '__main__':
    unittest.main()
