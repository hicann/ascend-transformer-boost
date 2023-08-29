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


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


OP_NAME = "MatmulOperation"
PARAM = '{"transposeA": false, "transposeB": false}'

class TestMatmul2_2Operation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.matmul(in_tensors[0], in_tensors[1])
        return [golden_result]

    def test(self):
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version in [104, 220, 221, 222, 223]:
            self.execute(OP_NAME, PARAM, 
                        [torch.rand(3, 4).npu().half(),
                        torch.rand(4, 5).npu().half()])
        else:
            print("TestMatmul2_2 310p skip")
            


if __name__ == '__main__':
    unittest.main()
