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


ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "examples/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)


class TestNormal(unittest.TestCase):
    def test_2d(self):
        operation = torch.classes.OperationTorch.OperationTorch()
        operation.test()
        a = torch.rand(384, 32, 1024).npu().half()
        b = torch.rand(1024, 1024).npu().half()
        c = torch.rand(1024).npu().half()

        if len(a.size()) == 3:
            result = torch.zeros(a.size()[0], a.size()[
                                 1], b.size()[0]).npu().half()
        else:
            result = torch.zeros(
                {a.size()[0], b.size()[0]}, a.options()).npu().half()
        print(result.size())
        operation.execute("LinearOperation", '{"transposeA":false, "transposeB":true}', [
                          a, b, c], [result])

        golden_result = torch.matmul(a, torch.transpose(b, 0, 1)) + c

        self.assertTrue(torch.allclose(
            result, golden_result, rtol=0.02, atol=0.02))


if __name__ == '__main__':
    unittest.main()
