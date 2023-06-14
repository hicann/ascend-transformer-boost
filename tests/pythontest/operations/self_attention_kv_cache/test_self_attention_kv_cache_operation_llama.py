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
from tensor_file import read_tensor  # NOQA: E402


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


OP_NAME = "SelfAttentionKvCacheOperation"
PARAM = '{"transKey": true, "headNum": 32, "layerId": 0, "dk": 128, "model": "llama7b"}'
INTENSOR0 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "inTensors0.path")
INTENSOR1 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "inTensors1.path")
INTENSOR2 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "inTensors2.path")
INTENSOR3 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "inTensors3.path")
INTENSOR4 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "inTensors4.path")
INTENSOR5 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "inTensors5.path")
OUTTENSOR0 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "outTensors0.path")
OUTTENSOR1 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "outTensors1.path")
OUTTENSOR2 = os.path.join("/home/fengrui/ascend-transformer-acceleration-fr/my_golden", "outTensors2.path")


class TestSelfAttentionKvCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        in0 = read_tensor(OUTTENSOR0).contiguous()
        # in0 = in0.view(in0.shape[0], in0.shape[1], in0.shape[2] * in0.shape[3])
        print("golden shape:")
        print(in0.shape)
        print(in0)
        print(read_tensor(OUTTENSOR1).shape)
        print(read_tensor(OUTTENSOR2).shape)
        # return [torch.rand(1, 1, 32, 128), torch.rand(1, 1, 32, 128), torch.rand(1, 1, 32, 128)]
        return [in0.npu(),
                read_tensor(OUTTENSOR1).npu(),
                read_tensor(OUTTENSOR2).npu()]

    def test(self):
        print(read_tensor(INTENSOR0).shape)
        print(read_tensor(INTENSOR1).shape)
        print(read_tensor(INTENSOR2).shape)
        print(read_tensor(INTENSOR3).shape)
        print(read_tensor(INTENSOR4).shape)
        print(read_tensor(INTENSOR5).shape)
        # q = torch.rand(1, 1, 32, 128)
        # k = torch.rand(1, 1, 32, 128)
        # v = torch.rand(1, 1, 32, 128)
        # am = torch.rand(1, 1, 2, 2)
        # pk = torch.rand(1, 1, 32, 128)
        # pv = torch.rand(1, 1, 32, 128)
        self.execute(OP_NAME, PARAM, [read_tensor(INTENSOR0).npu(),
                                      read_tensor(INTENSOR1).npu(),
                                      read_tensor(INTENSOR2).npu(),
                                      read_tensor(INTENSOR3).npu(),
                                      read_tensor(INTENSOR4).npu(),
                                      read_tensor(INTENSOR5).npu()])


if __name__ == '__main__':
    unittest.main()
