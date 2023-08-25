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


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


OP_NAME = "RmsNormQuantOperation"

input_scale = 12.0
input_offset = -20


class TestRmsNormQuantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        x, g, b = in_tensors
        x = x.cpu().numpy()
        g = g.cpu().numpy()
        b = b.cpu().numpy()

        num_row, num_col = x.shape[0], x.shape[1]

        rms = np.tile(np.sqrt(np.mean(x * x, axis=1)).reshape(num_row, 1), num_col)

        ref = x * g / rms
        ref = ref + b

        ref = np.reshape(ref, (-1, 1))
        ref = np.clip((ref * input_scale + input_offset), -128, 127)
        ref = torch.tensor(np.round(ref).astype(np.int8)).view(num_row, num_col)

        return [ref.npu()]

    def test_rms_norm_quant(self):
        self.execute(OP_NAME, {"inputScale": input_scale, "inputOffset": input_offset},
                     [torch.rand(2, 32).npu().half(), 
                      torch.rand(1, 32).npu().half(), 
                      torch.rand(1, 32).npu().half()])


if __name__ == '__main__':
    unittest.main()
