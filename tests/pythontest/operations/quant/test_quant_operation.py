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


OP_NAME = "QuantOperation"
shape1 = (1, 1, 4096)
input = np.random.uniform(low=-2, high=2, size=shape1).astype(np.float16)
input = torch.from_numpy(input).npu()
input_scale = float(82.96)
input_offset = int(-36)

class TestQuantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        x = np.array(in_tensors[0].cpu())
        ref = x * np.float16(input_scale) + np.float16(input_offset)
        ref = np.clip(ref, np.float16(-128.0), np.float16(127.0))
        
        ref = torch.tensor(np.round(ref).astype(np.int8)).reshape(shape1)
        return [ref.to(torch.int8).npu()]

    def test_2d_half(self):
        op = self.get_op(OP_NAME, { "input_scale" : input_scale, "input_offset" : input_offset})
        self.my_execute(op, [input])
    
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