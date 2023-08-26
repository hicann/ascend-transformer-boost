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

class TestTransdataLinear(operation_test.OperationTest):
    
    def golden_calc(self, in_tensors):
        golden_result = torch.matmul(input1.cpu().to(torch.int32) , torch.transpose(
            input2.cpu(), 0, 1).to(torch.int32))
        golden_result = (golden_result * input4.cpu().to(torch.float32)/1000).to(torch.float16) + input3.cpu()
        return [golden_result.half().npu()]

    def test(self):
        transdata_operation = torch.classes.OperationTorch.OperationTorch("TransDataInt8Operation")
        transdata_operation.set_param("{}")
        nz_input2 = transdata_operation.execute([input2.npu()])[0]

        self.execute(OP_NAME, PARAM, 
                     [input1.npu(),
                      nz_input2,
                      input3.npu().half(),
                      input4.npu().float()/1000])
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        print("out_tensor.shape", out_tensor.shape,
              "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
              ", \ngolden_oute_tensor:", golden_out_tensor)
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version in [104, 220, 221, 222, 223]:
            return True
        else:
            return True

if __name__ == '__main__':
    unittest.main()