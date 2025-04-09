# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
 
import os
import unittest
import time
import numpy as np
import torch
import torch_npu
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import op_test
 
OP_NAME = "FusedAddTopkDivOperation"
SUPPORT_DTYPE = [torch.float32, torch.float16, torch.bfloat16]
NORM_FLAG = [True, False]
 
def golden(input_x, input_add_num, group_num, group_topk, n, k, activate_type, is_norm, scale):
    input_x = input_x.to(torch.float32)
    input_add_num = input_add_num.to(torch.float32)
    # sigmoid
    m = torch.nn.Sigmoid()
    output_sig = m(input_x)
    # add
    input0 = torch.add(output_sig, input_add_num)
    # group_topk
    token_num, expert_num = input0.shape
    input0 = torch.reshape(input0, (token_num, group_num, expert_num // group_num))
    output = input0.clone()
    group_tensor = torch.topk(input0, n).values
    group_tensor = torch.sum(group_tensor, dim=-1)
    sort_index = torch.from_numpy(np.argsort(-group_tensor.numpy(), kind='stable')) 
    cols_to_use = torch.arange(group_topk, group_num, dtype=torch.long) 
    row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(cols_to_use.shape[0])
    col_indices = sort_index.index_select(1, cols_to_use).view(-1)
    output[row_indices, col_indices] = float(0)
    group_top_k_res = torch.reshape(output, (token_num, expert_num))
    # topk
    sort_res = torch.sort(group_top_k_res, descending=True, stable=True)
    # gather
    gather_res = torch.gather(output_sig, -1, sort_res.indices[:, 0:k])
    if is_norm:
        # reduce_sum
        sum_res = torch.sum(gather_res, -1, keepdim=True)
        # div
        res = torch.div(gather_res, sum_res)
        # mul
        res = res * torch.tensor(scale, dtype=torch.float32)
    else:
        res = gather_res
    return res, sort_res.indices[:, 0:k].to(torch.int32) 

class TestFusedAddTopkDiv(op_test.OpTest):
    def golden_calc(self, in_tensors):
        input_x = in_tensors[0].cpu()
        input_add_num = in_tensors[1].cpu()

        return golden(input_x, input_add_num, self.group_num, self.group_topk, self.n, 
                      self.k, self.activate_type, self.is_norm, self.scale)

    def golden_compare(self, out_tensors, golden_out_tensors):
        diff = torch.abs(torch.subtract(out_tensors[0].float(), golden_out_tensors[0].float()))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype),
                                   torch.abs(golden_out_tensors[0])).float()

        err_factor, eb_factor = 2**(-10), 2**(-14)

        if torch.any(torch.greater(diff, err_factor * tensor_max)):
            print("[new standards] output0 accuracy failed")
            return False

        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), eb_factor)):
            print("[new standards] output0 eb failed")
            return False

        diff = torch.abs(torch.subtract(out_tensors[1], golden_out_tensors[1]))
        if torch.any(torch.greater(diff, 1)):
            print("[new standards] output1 accuracy failed")
            return False

        return True

    @op_test.only_910b
    def test_fused_add_topk_div_model_case_16384_256(self):
        for dtype in SUPPORT_DTYPE:
            a = 16384
            b = 256
            self.group_num = 8
            self.group_topk = 4
            self.n = 2
            self.k = 8
            self.activate_type = 0
            for is_norm in NORM_FLAG:
                self.is_norm = is_norm
                self.scale = float(2.5)
                OP_PARAM0 = {"groupNum": self.group_num,"groupTopk": self.group_topk,"n": self.n,
                            "k": self.k,"activateType": self.activate_type,"isNorm": self.is_norm, "scale": self.scale}
        
                input_x_golden = torch.from_numpy(np.random.uniform(-10, 10, [a, b]).astype(np.float32)).to(dtype)
                input_add_num_golden = torch.from_numpy(np.random.uniform(-10, 10, b).astype(np.float32)).to(dtype)
                
                output_shape = [a, self.k]
                self.set_param(OP_NAME, OP_PARAM0)
                self.execute([input_x_golden, input_add_num_golden],
                            [torch.zeros(output_shape).to(torch.float32), torch.zeros(output_shape).to(torch.int32)])
    
    @op_test.only_910b
    def test_fused_add_topk_div_model_case_16_6(self):
        for dtype in SUPPORT_DTYPE:
            a = 16
            b = 6
            self.group_num = 3
            self.group_topk = 2
            self.n = 2
            self.k = 4
            self.activate_type = 0
            for is_norm in NORM_FLAG:
                self.is_norm = is_norm
                self.scale = float(2.5)
                OP_PARAM0 = {"groupNum": self.group_num,"groupTopk": self.group_topk,"n": self.n,
                            "k": self.k,"activateType": self.activate_type,"isNorm": self.is_norm, "scale": self.scale}
        
                input_x_golden = torch.from_numpy(np.random.uniform(-10, 10, [a, b]).astype(np.float32)).to(dtype)
                input_add_num_golden = torch.from_numpy(np.random.uniform(-10, 10, b).astype(np.float32)).to(dtype)
                
                output_shape = [a, self.k]
                self.set_param(OP_NAME, OP_PARAM0)
                self.execute([input_x_golden, input_add_num_golden],
                            [torch.zeros(output_shape).to(torch.float32), torch.zeros(output_shape).to(torch.int32)])
    
    @op_test.only_910b
    def test_fused_add_topk_div_model_case_16_16(self):
        for dtype in SUPPORT_DTYPE:
            a = 16
            b = 16
            self.group_num = 4
            self.group_topk = 2
            self.n = 2
            self.k = 8
            self.activate_type = 0
            for is_norm in NORM_FLAG:
                self.is_norm = is_norm
                self.scale = float(1.0)
                OP_PARAM0 = {"groupNum": self.group_num,"groupTopk": self.group_topk,"n": self.n,
                            "k": self.k,"activateType": self.activate_type,"isNorm": self.is_norm, "scale": self.scale}
        
                input_x_golden = torch.from_numpy(np.random.uniform(-10, 10, [a, b]).astype(np.float32)).to(dtype)
                input_add_num_golden = torch.from_numpy(np.random.uniform(-10, 10, b).astype(np.float32)).to(dtype)
                
                output_shape = [a, self.k]
                self.set_param(OP_NAME, OP_PARAM0)
                self.execute([input_x_golden, input_add_num_golden],
                            [torch.zeros(output_shape).to(torch.float32), torch.zeros(output_shape).to(torch.int32)])
    
    @op_test.only_910b
    def test_fused_add_topk_div_model_case_16_256(self):
        for dtype in SUPPORT_DTYPE:
            a = 16
            b = 256
            self.group_num=8
            self.group_topk=4
            self.n = 2
            self.k = 8
            self.activate_type=0
            for is_norm in NORM_FLAG:
                self.is_norm = is_norm
                self.scale=1.0
                OP_PARAM0 = {"groupNum": self.group_num,"groupTopk": self.group_topk,"n": self.n,
                            "k": self.k,"activateType": self.activate_type,"isNorm": self.is_norm, "scale": self.scale}
        
                input_x_golden = torch.from_numpy(np.random.uniform(-10, 10, [a, b]).astype(np.float16)).to(dtype)
                input_add_num_golden = torch.from_numpy(np.random.uniform(-10, 10, b).astype(np.float16)).to(dtype)
                
                output_shape = [a, self.k]
                self.set_param(OP_NAME, OP_PARAM0)
                self.execute([input_x_golden, input_add_num_golden],
                            [torch.zeros(output_shape).to(torch.float32), torch.zeros(output_shape).to(torch.int32)])
 
    @op_test.only_910b
    def test_fused_add_topk_div_fuzz_case_16_144(self):
        for dtype in SUPPORT_DTYPE:
            a = 16
            b = 144
            self.group_num = 8
            self.group_topk = 7
            self.n = 15
            self.k = 93
            self.activate_type = 0
            for is_norm in NORM_FLAG:
                self.is_norm = is_norm
                self.scale = 1.0
                OP_PARAM0 = {"groupNum": self.group_num,"groupTopk": self.group_topk,"n": self.n,
                            "k": self.k,"activateType": self.activate_type,"isNorm": self.is_norm, "scale": self.scale}
        
                input_x_golden = torch.from_numpy(np.random.uniform(-10, 10, [a, b]).astype(np.float16)).to(dtype)
                input_add_num_golden = torch.from_numpy(np.random.uniform(-10, 10, b).astype(np.float16)).to(dtype)
                
                output_shape = [a, self.k]
                self.set_param(OP_NAME, OP_PARAM0)
                self.execute([input_x_golden, input_add_num_golden],
                            [torch.zeros(output_shape).to(torch.float32), torch.zeros(output_shape).to(torch.int32)])
 
    @op_test.only_910b
    def test_fused_add_topk_div_fuzz_case_16_192(self):
        for dtype in SUPPORT_DTYPE:
            a = 16
            b = 192
            self.group_num = 8
            self.group_topk = 4
            self.n = 8
            self.k = 141
            self.activate_type = 0
            for is_norm in NORM_FLAG:
                self.is_norm = is_norm
                self.scale = 1.0
                OP_PARAM0 = {"groupNum": self.group_num,"groupTopk": self.group_topk,"n": self.n,
                            "k": self.k,"activateType": self.activate_type,"isNorm": self.is_norm, "scale": self.scale}
        
                input_x_golden = torch.from_numpy(np.random.uniform(-10, 10, [a, b]).astype(np.float16)).to(dtype)
                input_add_num_golden = torch.from_numpy(np.random.uniform(-10, 10, b).astype(np.float16)).to(dtype)
                
                output_shape = [a, self.k]
                self.set_param(OP_NAME, OP_PARAM0)
                self.execute([input_x_golden, input_add_num_golden],
                            [torch.zeros(output_shape).to(torch.float32), torch.zeros(output_shape).to(torch.int32)])

if __name__ == '__main__':
    unittest.main()
