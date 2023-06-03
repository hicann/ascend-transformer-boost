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
from tensor_testcase import TensorTestCase
import unittest
import os
import json
import torch
import torch_npu
import sys
sys.path.append('../..')

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "examples/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)


class TestBert(unittest.TestCase):
    def test_2d(self):
        operation = torch.classes.AddNormOperationTorch.AddNormOperationTorch()
        operation.test()
        testcase = TensorTestCase('AddLayerNorm', in_tensor_num=4)
        for i in range(1, 2):
            testcase.read(i)
            in_tensors = testcase.get_in_tensors()
            out_tensors = testcase.get_out_tensors()
            a = in_tensors[0].npu()
            b = in_tensors[1].npu()
            weight = in_tensors[2].npu()
            bias = in_tensors[3].npu()
            print(a.size())
            print(b.size())
            print(weight.size())
            print(bias.size())
            d = operation.execute(a, b, weight, bias)
            golden_d = out_tensors[0].npu()
            print("d:" + str(d.size()))
            print("golden_d:" + str(golden_d.size()))

            self.assertTrue(torch.allclose(d, golden_d, rtol=0.02, atol=0.02))


if __name__ == '__main__':
    unittest.main()
