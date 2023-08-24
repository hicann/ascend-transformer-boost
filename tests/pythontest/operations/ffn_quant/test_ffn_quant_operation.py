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


# sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# import operation_test  # NOQA: E402
sys.path.append(os.path.dirname(__file__))
import operation_test  # NOQA: E402


OP_NAME = "FfnQuantOperation"
PARAM = '{"transposeA": false, "transposeB": false}'
def padding_descale(x):
    zeros = torch.zeros(x.shape).npu()
    result = torch.cat(
        (x.unsqueeze(1).npu(), zeros.unsqueeze(1).npu()), dim=1).view(-1).npu()
    return result

class TetFfn(operation_test.OperationTest):
    
    def golden_calc(self, in_tensors):
        golden_result = torch.matmul(in_tensors[0].to(torch.float32) , torch.transpose(
            in_tensors[1], 0, 1).to(torch.float32))+ in_tensors[2].to(torch.float32)
        golden_result = (golden_result * in_tensors[3][::2].to(torch.float32)).to(torch.float16)
        return [torch.fast_gelu(golden_result)]

    def test(self):
        self.execute(OP_NAME, PARAM, 
                     [torch.randint(low=-128, high=127,size=(1,32, 1024),dtype=torch.int8).npu(),
                      torch.randint(low=-128, high=127,size=(4096, 1024),dtype=torch.int8).npu(),
                      torch.randint(low=-128, high=127,size=(4096,),dtype=torch.int).npu(),
                      padding_descale(torch.rand(4096).npu().float()/1000)])

if __name__ == '__main__':
    unittest.main()
