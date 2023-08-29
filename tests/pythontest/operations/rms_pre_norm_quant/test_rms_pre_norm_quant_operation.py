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


OP_NAME = "RmsPreNormQuantOperation"

input_scale = 12.0
input_offset = -20
epsilon = 1e-6
QUANT_MAX = 127.0
QUANT_MIN = -128.0


class TestRmsPreNormQuantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        x, resin, g, b = in_tensors
        x = x.squeeze(0).cpu().numpy().astype(np.float32)
        resin = resin.squeeze(0).cpu().numpy().astype(np.float32)
        g = g.cpu().numpy().astype(np.float32)
        b = b.cpu().numpy().astype(np.float32)
        num_row, num_col = x.shape[0], x.shape[1]
        x = x + resin
        out1 = torch.tensor(x.astype(np.float16)).reshape(1, num_row, num_col)
        rms = np.tile(np.sqrt(np.mean(x * x, axis=1)).reshape(num_row, 1), num_col)
        rms = rms + epsilon
        ref = x * g / rms
        ref = ref + b
        ref = np.reshape(ref, (-1, 1))
        ref = ref * np.float32(input_scale) + np.float32(input_offset)
        ref = np.clip(ref, np.float32(QUANT_MIN), np.float32(QUANT_MAX))
        ref = ref.astype(np.float16)
        ref = torch.tensor(np.round(ref).astype(np.int8)).reshape(1, num_row, num_col)

        return [ref.npu(), out1.npu()]

    def test_rms_pre_norm_quant(self):
        self.execute(OP_NAME, {"inputScale": input_scale, "inputOffset": input_offset, "rmsNormEps" : epsilon},
                     [torch.rand(1, 2, 32).npu().half(), 
                      torch.rand(1, 2, 32).npu().half(), 
                      torch.rand(1, 1, 32).npu().half(),
                      torch.rand(1, 1, 32).npu().half()])

    def golden_compare(self, out_tensor, golden_out_tensor):
        # print("out_tensor.shape", out_tensor.shape,
        #       "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        # print("out_tensor:", out_tensor,
        #       ", \ngolden_oute_tensor:", golden_out_tensor)
        max_error = torch.max(torch.abs(out_tensor - golden_out_tensor))
        # print("max error", max_error)
        if max_error <= 1:
            return True
        else:
            return False

if __name__ == '__main__':
    unittest.main()
