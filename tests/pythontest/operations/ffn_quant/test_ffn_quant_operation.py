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

torch.manual_seed(1234)

OP_NAME = "FfnQuantOperation"
PARAM = '{"transposeA": false, "transposeB": false}'
def padding_descale(x):
    zeros = torch.zeros(x.shape).npu()
    result = torch.cat(
        (x.unsqueeze(1).npu(), zeros.unsqueeze(1).npu()), dim=1).view(-1).npu()
    return result
#data
input1 = torch.randint(low=-128, high=127,size=(1,32, 1024),dtype=torch.int8).npu()
input2 = torch.randint(low=-128, high=127,size=(4096, 1024),dtype=torch.int8).npu()
input3 = torch.rand(4096).npu().half()
input4 = torch.rand(4096).npu().float()/1000
if soc_version in [104, 220, 221, 222, 223]:
    input1 = torch.randint(low=-128, high=127,size=(1,32, 1024),dtype=torch.int8).npu()
    input2 = torch.randint(low=-128, high=127,size=(4096, 1024),dtype=torch.int8).npu()
    input3 = torch.randint(low=-128, high=127,size=(4096,),dtype=torch.int).npu()
    input4 = padding_descale(torch.rand(4096).npu().float()/1000)

class TestFfnQuantOperation(operation_test.OperationTest):
    
    def golden_calc(self, in_tensors):
        golden_result = torch.randint(low=-128, high=127,size=(1,32, 1024),dtype=torch.int8)
        if soc_version in [104, 220, 221, 222, 223]:
            golden_result = torch.matmul(in_tensors[0].to(torch.float32) , torch.transpose(
                in_tensors[1], 0, 1).to(torch.float32))+ in_tensors[2].to(torch.float32)
            golden_result = (golden_result * in_tensors[3][::2].to(torch.float32)).to(torch.float16)
        else:
            golden_result = torch.matmul(in_tensors[0].cpu().to(torch.int32) , torch.transpose(
                in_tensors[1].cpu(), 0, 1).to(torch.int32))
            golden_result = (golden_result * in_tensors[3].cpu().to(torch.float32)).to(torch.float16) + in_tensors[2].cpu()
        return [torch.fast_gelu(golden_result.npu())]

    def test(self):
        self.execute(OP_NAME, PARAM, 
                     [input1,
                      input2,
                      input3,
                      input4])

if __name__ == '__main__':
    unittest.main()
