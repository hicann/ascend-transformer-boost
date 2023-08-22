# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import unittest
import torch
import torch_npu
import numpy as np

sys.path.append(os.path.dirname(__file__))
import operation_test  # NOQA: E402
torch.npu.set_device(torch.device("npu:0"))

OP_NAME = "NormQuantOperation"

input_scale = 82.96
input_offset = -36
alpha_val = 0.5
shape1 = (1, 1, 4096)
shape2 = (4096)
input_data = torch.from_numpy(np.random.uniform(low=-2, high=2, size=shape1).astype(np.float16)).npu()
weight_data = torch.from_numpy(np.random.uniform(low=-2, high=2, size=shape2).astype(np.float16)).npu()
bias_data = torch.from_numpy(np.random.uniform(low=-2, high=2, size=shape2).astype(np.float16)).npu()

def layer_norm_compute(input_x, input_gamma, input_beta,
                       begin_norm_axis, begin_params_axis,
                       epsilon, impl_mode="high_performance"):

    shape_x = input_x.shape
    dtype = input_x.dtype
    if dtype == "float16":
        input_x = input_x.astype("float32")
        input_gamma = input_gamma.astype("float32")
        input_beta = input_beta.astype("float32")

    index_list = tuple(index for index, _ in enumerate(shape_x))
    reduce_axis = index_list[begin_norm_axis:]

    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    mean_cof = reduce_elts ** (-1)

    if impl_mode != "keep_fp16":
        mean_muls = input_x * mean_cof
        
        mean = np.sum(mean_muls, axis=reduce_axis, keepdims=True)
        mean_variance_broadcast = np.broadcast_to(mean, shape_x)
        variance_sub = np.subtract(input_x, mean_variance_broadcast)
        variance_mul = np.multiply(variance_sub, variance_sub)
        variance_muls = variance_mul * mean_cof
        variance = np.sum(variance_muls, axis=reduce_axis, keepdims=True)

    if impl_mode == "high_performance":
        mean_normalize_broadcast = np.broadcast_to(mean, shape_x)
        
        normalize_sub = np.subtract(input_x, mean_normalize_broadcast)
        variance_normalize_broadcast = np.broadcast_to(variance, shape_x)
        normalize_add = variance_normalize_broadcast + epsilon
        normalize_log = np.log(normalize_add)
        normalize_log_mul = normalize_log * (-0.5)
        normalize_exp = np.exp(normalize_log_mul)
        normalize_mul = np.multiply(normalize_sub, normalize_exp)
    if begin_params_axis == 0:
        scale_mul = np.multiply(input_gamma, normalize_mul)
        res = np.add(scale_mul, input_beta)
    else:
        gamma_broadcast = np.broadcast_to(input_gamma, shape_x)
        print(gamma_broadcast.shape)
        print(normalize_mul.shape)
        beta_broadcast = np.broadcast_to(input_beta, shape_x)
        scale_mul = np.multiply(gamma_broadcast, normalize_mul)
        res = np.add(scale_mul, beta_broadcast)

    resout = res.astype("float16")
    resout = resout * alpha_val
    res = res.astype("float16")
    res = res * input_scale + input_offset
    res = np.clip(res, -128, 127)
    res = np.round(res)

    if dtype == "float16":
        mean = mean.astype("float16")
        variance = variance.astype("float16")
        res = res.astype("int8")

    return res, resout

class TestAddNormQuantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        quant_result, golden_result = layer_norm_compute(np.array(in_tensors[0].cpu()).astype(np.float16), 
                                                         np.array(in_tensors[1].cpu()).astype(np.float16), 
                                                         np.array(in_tensors[2].cpu()).astype(np.float16), 
                                                         1, 1, 1e-12)
        return [torch.tensor(quant_result).npu(), torch.tensor(golden_result).npu()]

    def test_2d_half(self):
        self.execute(OP_NAME, {"layerNormEps": 1e-12, 
                               "input_scale" : input_scale, 
                               "input_offset" : input_offset, 
                               "input_alpha" : alpha_val}, 
                     [input_data.npu().half(), 
                      weight_data.npu().half(), 
                      bias_data.npu().half()])

    def golden_compare(self, out_tensor, golden_out_tensor):
        print("out_tensor.shape", out_tensor.shape,
              "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
              ", \ngolden_oute_tensor:", golden_out_tensor)
        max_error = torch.max(torch.abs(out_tensor - golden_out_tensor))
        print("max error", max_error)
        if max_error <= 1:
            return True
        else:
            return False

if __name__ == '__main__':
    unittest.main()
