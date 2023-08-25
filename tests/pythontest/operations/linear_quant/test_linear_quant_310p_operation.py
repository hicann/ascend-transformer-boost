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
NUM_BATCH = 2
BLOCK_SIZE_16 = 16
BLOCK_SIZE_32 = 32
TRANSPOSE_B = True


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "LinearQuantOperation"
PARAM = '{"transposeA": false, "transposeB": false}'

torch.manual_seed(1234)

input1 = torch.randint(low=-128, high=127,size=(1,32, 1024),dtype=torch.int8)
input2 = torch.randint(low=-128, high=127,size=(4096, 1024),dtype=torch.int8)
input3 = torch.rand(4096)
input4 = torch.rand(4096)

class TetFfn(operation_test.OperationTest):
    
    def golden_calc(self, in_tensors):
        golden_result = torch.matmul(in_tensors[0].cpu().to(torch.int32) , torch.transpose(
            in_tensors[1].cpu(), 0, 1).to(torch.int32))
        golden_result = (golden_result * in_tensors[3].cpu().to(torch.float32)).to(torch.float16) + in_tensors[2].cpu()
        return [golden_result.npu()]

    def test(self):
        nz_input2 = input2.npu().to(torch.float16).npu_format_cast(29).to(torch.int8)
        import pdb
        pdb.set_trace()
        self.execute(OP_NAME, PARAM, 
                     [input1.npu(),
                      nz_input2,
                      input3.npu().half(),
                      input4.npu().float()/1000])

if __name__ == '__main__':
    unittest.main()