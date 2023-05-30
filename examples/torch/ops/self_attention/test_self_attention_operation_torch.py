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
import unittest
import os
import json
import torch
import torch_npu
import sys
sys.path.append('../..')
from tensor_testcase import TensorTestCase

ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "examples/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)


# class TestNormal(unittest.TestCase):
#     def test_2d(self):
#         param = '{"transKey":true,"dk":64,"headNum":16}'
#         operation = torch.classes.SelfAttentionOperationTorch.SelfAttentionOperationTorch(param)
#         operation.test()
#         query = torch.rand(32, 384, 1024).npu()
#         key = torch.rand(32, 384, 1024).npu()
#         value = torch.rand(32, 384, 1024).npu()
#         mask = torch.rand(32, 1, 1, 384).npu()
#         result = operation.execute(query, key, value, mask)
#         print("result:" + str(result))


class TestBert(unittest.TestCase):
    def test_2d(self):
        param = '{"transKey":false,"dk":64,"headNum":16}'
        operation = torch.classes.SelfAttentionOperationTorch.SelfAttentionOperationTorch(param)
        operation.test()
        testcase = TensorTestCase('BertSelfAttention', in_tensor_num=7, out_tensor_num=6)
        testcase.read(1)
        in_tensors = testcase.get_in_tensors()
        out_tensors = testcase.get_out_tensors()
        query = in_tensors[4].npu()
        key = in_tensors[5].npu()
        value = in_tensors[6].npu()
        mask = in_tensors[3].npu()
        print(query.size())
        print(key.size())
        print(value.size())
        print(mask.size())
        d = operation.execute(query, key, value, mask)
        # d = d.view(32, 384, 1024)
        golden_d = out_tensors[0].npu()
        print("d:" + str(d.size()))
        print("golden_d:" + str(golden_d.size()))
        print("d:" + str(d))
        print("golden_d:" + str(golden_d))

        self.assertTrue(torch.allclose(d, golden_d, rtol=0.02, atol=0.02))


if __name__ == '__main__':
    unittest.main()
