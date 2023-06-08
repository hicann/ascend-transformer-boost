# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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


# class TestNormal(unittest.TestCase):
#     def test_2d(self):
#         param = '{"transposeA":false,"transposeB":true}'
#         operation = torch.classes.LinearOperationTorch.LinearOperationTorch(param)
#         a = torch.rand(5584, 1024).npu()
#         b = torch.rand(4096, 1024).npu()
#         c = torch.rand(4096).npu()
#         print(a.size())
#         print(b.size())
#         print(c.size())
#         d = operation.execute(a, b, c)
#         golden_d = torch.matmul(a, torch.transpose(b, 0, 1)) + c
#         print("d:" + str(d.size()))
#         print("golden_d:" + str(golden_d.size()))

#         self.assertTrue(torch.allclose(d, golden_d, rtol=0.02, atol=0.02))


class TestBert(unittest.TestCase):
    def test_2d(self):
        param = '{"transposeA":false,"transposeB":true}'
        operation = torch.classes.OperationTorch.OperationTorch(
            "LinearOperation")
        operation.set_param(param)
        testcase = TensorTestCase(
            'FastUnpadBertSelfAttention', in_tensor_num=12, out_tensor_num=6)
        for i in range(1, 2):
            testcase.read(i)
            in_tensors = testcase.get_in_tensors()
            out_tensors = testcase.get_out_tensors()
            a = in_tensors[0].npu()
            b = in_tensors[6].npu()
            c = in_tensors[7].npu()
            print(a.size())
            print(b.size())
            print(c.size())
            d = operation.execute(a, b, c)
            golden_d = out_tensors[0].npu()
            print("d:" + str(d.size()))
            print("golden_d:" + str(golden_d.size()))

            self.assertTrue(torch.allclose(d, golden_d, rtol=0.02, atol=0.02))


if __name__ == '__main__':
    unittest.main()
