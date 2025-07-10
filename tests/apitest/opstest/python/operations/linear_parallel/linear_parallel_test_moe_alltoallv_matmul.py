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

def get_err_threshold_for_one_golden(dtype:torch.dtype):
    if dtype == torch.float32:
        dtype = torch.float16
    if dtype in [torch.float16]:
        precision_threshold = 2 ** (-8)
        eb_threshold = 2 ** (-10)
    if dtype in [torch.bfloat16]:
        precision_threshold = 2 ** (-7)
        eb_threshold = 2 ** (-7)
    return precision_threshold

def get_err_threshold_for_two_golden(dtype:torch.dtype):
    if dtype in [torch.bfloat16]:
        err_threshold = 2 ** (-8)
    if dtype in [torch.float16]:
        err_threshold = 2 ** (-11)
    if dtype in [torch.float32]:
        err_threshold = 2 ** (-14)
    return err_threshold

def get_eb_threshold(dtype:torch.dtype):
    eb_threshold = 0
    if dtype in [torch.bfloat16]:
        eb_threshold = 2**(-7)
    if dtype in [torch.float16]:
        eb_threshold = 2**(-10)
    if dtype in [torch.float32]:
        eb_threshold = 2**(-14)
    return eb_threshold

def get_eb(golden:torch.Tensor, actual:torch.Tensor):
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    actual_error = actual - golden
    EB = torch.mean(actual_error / golden_nmax)
    return EB

def one_golden_compare(tensor_a, tensor_b):
    err = get_err_threshold_for_one_golden(tensor_a.dtype)
    if torch.isnan(tensor_a).any():
        print("********Warning: npu result contains NaN!*************")
        return 1
    tensor_a = tensor_a.to(torch.float32)
    tensor_b = tensor_b.to(torch.float32)
    # 确定性计算要求2次npu计算结果完全一致
    if os.getenv('LCCL_DETERMINISTIC', '0') == "1":
        if torch.equal(tensor_a, tensor_b):
            return 0
        return 1
    golden_nmax = torch.clamp(torch.abs(tensor_b), min = 1)
    abs_error = torch.abs(tensor_a - tensor_b)
    result = (abs_error <= err * golden_nmax).all()
    if result:
        return 0
    else:
        output_error_item(tensor_a, tensor_b)
        return 1


def main_worker(rank, comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
                data_type, quant_info, EP, TP, quant_type, out_data_ype):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')

    acl_alltoall_matmul_operation = torch.classes.OperationTorch.OperationTorch(
        "LinearParallelOperation")

    acl_param = json.dumps({"type": 5, "rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "transWeight": bool(trans_b), "backend": "lcoc",
                            "quantType": quant_type, "outDataType": out_data_ype,
                            "moeInfo": {"epSize": world_size, "localExpertNums":
                                local_expert_nums, "tpSize": 1}})

    acl_alltoall_matmul_operation.set_param(acl_param)
    torch.manual_seed(0)
    moedata = MoeTestDate(rank, CommType(comm_type), world_size, batch, M, K, N, trans_b, local_expert_nums,
                          CoCDataTypeDesc(data_type), quant_info, EP, TP)

    in_tensors = []
    input_tensor = moedata.matrix_a
    input_tensor = input_tensor.reshape(M, K)
    in_tensors.append(input_tensor.to(torch.device('npu')))

    weight_tensor = moedata.matrix_b
    if trans_b:
        weight_tensor = weight_tensor.reshape(local_expert_nums, N, K)
    else:
        weight_tensor = weight_tensor.reshape(local_expert_nums, K, N)
    in_tensors.append(weight_tensor.to(torch.device('npu')))
    if quant_type == 3:
        dequantScale = moedata.matrix_dequant_scale
        dequantScale = dequantScale.reshape(N * local_expert_nums)
        in_tensors.append(dequantScale.to(torch.device('npu')))

        quantScale = moedata.matrix_quant_scale
        quantScale = quantScale.reshape(M)
        in_tensors.append(quantScale.to(torch.device('npu')))
    elif quant_type == 1:
        dequantScale = moedata.matrix_dequant_scale
        dequantScale = dequantScale.reshape(N * local_expert_nums)
        in_tensors.append(dequantScale.to(torch.device('npu')))

    global_tokens_per_expert_matrix = moedata.global_tokens_per_expert_matrix
    in_tensors.append(global_tokens_per_expert_matrix.to(torch.device('npu')))

    maxOutputSize = torch.zeros(input_tensor.shape[0] * 2, dtype=torch.int32)
    in_tensors.append(maxOutputSize.to(torch.device('npu')))

    out_tensor = acl_alltoall_matmul_operation.execute(in_tensors)

    torch.npu.synchronize()
    golden_out_tensor = moedata.matrix_c
    golden_out_tensor_low = moedata.matrix_c_low
    out_tensor_compare = out_tensor[0].to(torch.device('cpu'))[:golden_out_tensor.shape[1], :]

    assert check_precision_new(out_tensor_compare, golden_out_tensor, golden_out_tensor_low)


def check_precision_new(tensor_a, tensor_b, tensor_c):
    if torch.isnan(tensor_a).any():
        print("********Warning: npu result contains NaN!*************")
        return 1
    epsilon = 1e-7
    d_type = tensor_a.dtype
    err_threshold = get_err_threshold_for_two_golden(d_type)
    eb_threshold = get_eb_threshold(d_type)

    tensor_a = tensor_a.to(torch.float32)
    tensor_b = tensor_b.to(torch.float32)
    tensor_c = tensor_c.to(torch.float32)

    relative_error_npu = torch.abs(tensor_a - tensor_b) / (torch.abs(tensor_b) + epsilon)
    relative_error_cpu = torch.abs(tensor_c - tensor_b) / (torch.abs(tensor_b) + epsilon)
    max_relative_error_npu = torch.max(relative_error_npu)
    max_relative_error_cpu = torch.max(relative_error_cpu)
    mean_relative_error_npu = torch.mean(relative_error_npu)
    mean_relative_error_cpu = torch.mean(relative_error_cpu)
    # 计算均方根误差
    mse_npu = torch.mean((tensor_a - tensor_b) ** 2)
    rmse_npu = torch.sqrt(mse_npu)
    mse_cpu = torch.mean((tensor_c - tensor_b) ** 2)
    rmse_cpu = torch.sqrt(mse_cpu)

    EB = torch.abs(get_eb(tensor_b, tensor_a))

    print("最大相对误差npu:", max_relative_error_npu)
    print("最大相对误差cpu:", max_relative_error_cpu)
    print("平均相对误差npu:", mean_relative_error_npu)
    print("平均相对误差cpu:", mean_relative_error_cpu)
    print("均方根误差npu:", rmse_npu)
    print("均方根误差cpu:", rmse_cpu)
    print("误差均衡性EB:", EB)

    if max_relative_error_npu / max(max_relative_error_cpu, err_threshold) >= 10:
        if one_golden_compare(tensor_a, tensor_b):
            print("resule is error")
            return 0

    if mean_relative_error_npu / max(mean_relative_error_cpu, err_threshold) >= 2 or rmse_npu / max(rmse_cpu, err_threshold) >= 2 or EB >= eb_threshold:
        print("result is error")
        return 0
    print("result is same with expect")
    return 1

class LinearParallelCoverOperationTest(operation_test.OperationTest):

    # def test_linear_paraller_fp16(self):
    #     if not operation_test.get_soc_version() == 'Ascend910B':
    #         return
    #     print(f"———————— LinearParallelCoverOp test start ————————")
    #     print("------------ALLTOALLVC ALLGATHER MATMUL Non quantitative scenarios-----------")
    #     world_size = 4
    #     comm_type = 309
    #     batch = 1
    #     M = 40060
    #     K = 2134
    #     N = 1565
    #     trans_b = 1
    #     quant_granularity = -1
    #     quant_group_size = -1
    #     has_quant_offset = -1
    #     dequant_group_size = -1
    #     local_expert_nums = 11
    #     EP = 4
    #     TP = 1
    #     out_data_type = 27
    #     dequant_granularity = -1
    #     has_dequant_offset = -1
    #     data_type = 1
    #     quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
    #                            QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)
    #     mp.spawn(main_worker, nprocs=world_size,
    #              args=(comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
    #                    CoCDataTypeDesc(data_type), quant_info, EP, TP, dequant_granularity, out_data_type))

    # def test_linear_paraller_fp16(self):
    #     if not operation_test.get_soc_version() == 'Ascend910B':
    #         return
    #     print(f"———————— LinearParallelCoverOp test start ————————")
    #     print("------------ALLTOALLVC ALLGATHER MATMUL Non quantitative scenarios-----------")
    #     world_size = 8
    #     comm_type = 309
    #     batch = 1
    #     M = 783
    #     K = 8075
    #     N = 2455
    #     trans_b = 0
    #     quant_granularity = -1
    #     quant_group_size = -1
    #     has_quant_offset = -1
    #     dequant_group_size = -1
    #     local_expert_nums = 2
    #     EP = 8
    #     TP = 1
    #     out_data_type = 27
    #     dequant_granularity = -1
    #     has_dequant_offset = -1
    #     data_type = 1
    #     quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
    #                            QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)
    #     mp.spawn(main_worker, nprocs=world_size,
    #              args=(comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
    #                    CoCDataTypeDesc(data_type), quant_info, EP, TP, dequant_granularity, out_data_type))

    # def test_linear_paraller_fp16_quant(self):
    #     if not operation_test.get_soc_version() == 'Ascend910B':
    #         return
    #     print(f"———————— LinearParallelCoverOp test start ————————")
    #     print("------------ALLTOALLVC ALLGATHER MATMUL Quantify scenarios-----------")
    #     world_size = 8
    #     comm_type = 309
    #     batch = 1
    #     M = 3125
    #     K = 6220
    #     N = 8692
    #     trans_b = 1
    #     quant_granularity = -1
    #     quant_group_size = -1
    #     has_quant_offset = -1
    #     dequant_group_size = -1
    #     local_expert_nums = 8
    #     EP = 8
    #     TP = 1
    #     out_data_type = 1
    #     dequant_granularity = 1
    #     has_dequant_offset = -1
    #     data_type = 2
    #     quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
    #                            QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)
    #     mp.spawn(main_worker, nprocs=world_size,
    #              args=(comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
    #                    CoCDataTypeDesc(data_type), quant_info, EP, TP, dequant_granularity, out_data_type))

    def test_linear_paraller_fp16(self):
        for i in range(100):
            if not operation_test.get_soc_version() == 'Ascend910B':
                return
            print(f"———————— LinearParallelCoverOp test start ————————")
            print("------------ALLTOALLVC ALLGATHER MATMUL Non quantitative scenarios-----------")
            # world_size = 8 # 2的幂次
            import random
            world_size = random.choice([4, 8])  # 2的幂次
            # world_size = random.choice([2, 4, 8, 16]) # 2的幂次
            comm_type = 309

            def generate_batch(num):
                low_range = list(range(1, 10000))
                middle_range = list(range(10000, 30000))
                high_range = list(range(30000, 32768))
                candidates = low_range + high_range + middle_range
                weights = [85 / 10000] * len(low_range) + [10 / 20000] * len(middle_range) + [5 / 72400] * len(
                    high_range)
                batch = random.choices(candidates, weights=weights, k=1)[0]
                return batch

            batch = 1
            M = 1024  #
            M = generate_batch(100)
            # K = 1024 # k 7168
            K = generate_batch(128)
            # N = 1024 # n 4096
            N = random.randint(1, 10000)
            while M > 200 * 1024:
                M = generate_batch(100)
            trans_b = random.randint(0, 1)
            quant_granularity = -1
            quant_group_size = -1
            has_quant_offset = -1
            dequant_group_size = -1
            # local_expert_nums = 4 # 1- 16
            local_expert_nums = random.randint(1, 16)  # 1- 16
            # EP = 8 # EP * TP = WORLDSIZE
            EP = world_size
            while EP > world_size:
                EP = random.choice([1, 2, 4, 8, 16])
            TP = world_size // EP
            # TP = 1 # 客户场景 TP=1
            # out_data_type = 1 # 1对应data_type=0 27对应data_type=1

            # dequant_granularity = -1  # 量化类型 -1 非量化
            dequant_granularity = random.choice([-1, 1, 3])  # 量化类型 -1 非量化
            if dequant_granularity == -1:
                out_data_type = random.choice([1, 27])
                data_type = 0 if out_data_type == 1 else 1
            else:
                out_data_type = 1
                data_type = 2
            print(
                f"--M:{M}--N:{N}--K:{K}--world_size:{world_size}--local_expert_nums:{local_expert_nums}--EP:{EP}--TP:{TP}--dequant_granularity:{dequant_granularity}--out_data_type:{out_data_type}--data_type:{data_type}")
            has_dequant_offset = -1

            quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
                                QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)
            mp.spawn(main_worker, nprocs=world_size,
                    args=(comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
                        CoCDataTypeDesc(data_type), quant_info, EP, TP, dequant_granularity, out_data_type))

if __name__ == '__main__':
    unittest.main()

