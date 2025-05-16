#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import torch
import torch_npu

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test
from precision_calcu import compare_cv

OP_NAME = "GmmDeqSwigluQuantGmmDeqOperation"

def swiglu(x: torch.Tensor) -> torch.Tensor:
    x0, gate = x.chunk(2, dim=-1)
    swish = x0 * torch.sigmoid(x0)
    y = swish * gate
    return y

def permute_weight(w: torch.Tensor, tile_n: int):
    *dims, n = w.shape
    order = list(range(len(dims))) + [-2, -3, -1]
    return w.reshape(*dims, 2, n // tile_n, tile_n // 2).permute(order).reshape(*dims, n)

def random_uniform(lower: float, upper: float, size: tuple, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(size=size, dtype=dtype) * (upper - lower) + lower

# 真值计算
def generate_data(group_list, m, n, k, calc_dtype=torch.float32):
    group_num = len(group_list)
    m_actual = group_list[-1]
    n_out, n2, k2 = n // 2, k, n // 2

    # 数据生成
    x1 = torch.randint(-16, 16, size=(m, k), dtype=torch.int8)
    weight1 = torch.randint(-16, 16, size=(group_num, k, n), dtype=torch.int8)
    scale1 = random_uniform(0.004, 0.005, size=(group_num, n,), dtype=torch.float32)
    per_token_scale1 = random_uniform(0.004, 0.005, size=(m,), dtype=torch.float32)
    group_list = torch.Tensor(group_list).to(torch.int64)
    weight2 = torch.randint(-16, 16, size=(group_num, k2, n2,), dtype=torch.int8)
    scale2 = random_uniform(0.004, 0.005, size=(group_num, n2,), dtype=torch.float32)
    # 中间参数
    x2 = torch.empty(size=(m, k2), dtype=torch.int8)
    per_token_scale2 = torch.empty(size=(m,), dtype=torch.float32)

    # Grouped matmul dequant
    matmul_result = torch.empty(size=(m_actual, n), dtype=calc_dtype)
    for group_idx in range(group_num):
        start = 0 if group_idx == 0 else group_list[group_idx - 1]
        end = group_list[group_idx]
        res = torch.matmul(
            x1[start:end, :].to(torch.float32),
            weight1[group_idx, :, :].to(torch.float32)
        ).to(calc_dtype)
        res *= scale1[group_idx, :].to(calc_dtype)
        res *= per_token_scale1[start:end, None].to(calc_dtype)
        matmul_result[start:end, :] = res

    # 检查 matmul result
    if not torch.isfinite(matmul_result.to(torch.float16)).all():
        return None

    # Swiglu
    swiglu_out = swiglu(matmul_result)

    # 检查 swiglu_out
    if not torch.isfinite(swiglu_out.to(torch.float16)).all():
        return None

    # Quant
    out_max = torch.max(torch.abs(swiglu_out), dim=-1).values
    quant_result = swiglu_out * 127. / out_max[:, None]
    x2[:m_actual] = torch.round(quant_result).to(torch.int8)
    per_token_scale2[:m_actual] = (out_max / 127.).to(torch.float32)

    # 检查 x2 和 per_token_scale2
    if not torch.isfinite(x2).all() or not torch.isfinite(per_token_scale2).all():
        return None

    # Grouped matmul dequant
    true_value = torch.empty(size=(m, n2), dtype=calc_dtype)
    for group_idx in range(group_num):
        start = 0 if group_idx == 0 else group_list[group_idx - 1]
        end = group_list[group_idx]
        res = torch.matmul(
            x2[start:end, :].to(torch.float32),
            weight2[group_idx, :, :].to(torch.float32)
        ).to(calc_dtype)
        res *= scale2[group_idx, :].to(calc_dtype)
        res *= per_token_scale2[start:end, None].to(calc_dtype)
        true_value[start:end, :] = res

    # 检查 true_value
    if not torch.isfinite(true_value.to(torch.float16)).all():
        return None

    return (x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2, true_value)

def generate_data_safe(group_list, m, n, k, calc_dtype=torch.float32):
    max_attempt_times = 5
    for _ in range(max_attempt_times):
        data = generate_data(group_list, m, n, k, calc_dtype)
        if data is not None:
            return data
    raise ValueError(f"Try {max_attempt_times} times to generate data but still get illegal value.")

def calculate_golden(x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2):
    [out1] = torch_npu.npu_grouped_matmul(
        x=[x1], weight=[weight1], scale=[scale1], per_token_scale=[per_token_scale1],
        group_list=group_list, split_item=2, group_type=0, group_list_type=0, act_type=0, output_dtype=torch.float16
    )

    swiglu_out = torch_npu.npu_swiglu(input=out1, dim=-1)

    x2, per_token_scale2 = torch_npu.npu_dynamic_quant(input=swiglu_out, dst_type=torch.int8)

    [out2] = torch_npu.npu_grouped_matmul(
        x=[x2], weight=[weight2], scale=[scale2], per_token_scale=[per_token_scale2],
        group_list=group_list, split_item=2, group_type=0, group_list_type=0, act_type=0, output_dtype=torch.float16
    )

    torch.npu.synchronize()
    return out2

class TestGmmDeqSwigluQuantGmmDeq(operation_test.OperationTest):
    def golden_calc(self, in_tensor):
        return [self.golden]

    def golden_compare(self, out_tensor, golden_out_tensor):
        output, golden = out_tensor[:self.m_actual].cpu(), golden_out_tensor[:self.m_actual].cpu()
        true_value = self.true_value[:self.m_actual]

        return compare_cv(true_value, golden, output)

    def test_gmm_deq_swiglu_quant_gmm_deq_1(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "groupListType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        group_list = [432, 496]
        m, n, k = 512, 4096, 7168
        self.m_actual = group_list[-1]

        (x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2,
            self.true_value) = generate_data_safe(group_list, m, n, k)

        self.golden = calculate_golden(x1.npu(), weight1.npu(), scale1.npu(), per_token_scale1.npu(), group_list.npu(),
            weight2.npu(), scale2.npu())

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            group_list.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

    def test_gmm_deq_swiglu_quant_gmm_deq_2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "groupListType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        group_list = [360]
        m, n, k = 2464, 4096, 7168
        self.m_actual = group_list[-1]

        (x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2,
            self.true_value) = generate_data_safe(group_list, m, n, k)

        self.golden = calculate_golden(x1.npu(), weight1.npu(), scale1.npu(), per_token_scale1.npu(), group_list.npu(),
            weight2.npu(), scale2.npu())

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            group_list.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

    def test_gmm_deq_swiglu_quant_gmm_deq_3(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "groupListType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        group_list = [128, 256, 384]
        m, n, k = 2464, 4096, 7168
        self.m_actual = group_list[-1]

        (x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2,
            self.true_value) = generate_data_safe(group_list, m, n, k)

        self.golden = calculate_golden(x1.npu(), weight1.npu(), scale1.npu(), per_token_scale1.npu(), group_list.npu(),
            weight2.npu(), scale2.npu())

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            group_list.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

if __name__ == '__main__':
    unittest.main()
