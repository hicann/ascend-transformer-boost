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


class TestAddNormal(unittest.TestCase):
    def test_2d(self):
        operation = torch.classes.OperationTorch.OperationTorch(
            "NormOperation")
        operation.set_param(json.dumps({"layerNormEps": 1e-12}))
        a = torch.rand(2, 3).npu().half()
        normWeight = torch.rand(3).npu().half()
        normBias = torch.rand(3).npu().half()

        results = operation.execute([a, normWeight, normBias])

        layer_norm = torch.nn.LayerNorm([3]).npu()
        layer_norm.load_state_dict({"weight": normWeight, "bias": normBias})
        golden_result = layer_norm(a)

        print("result:" + str(results[0]))
        print("golden_result:" + str(golden_result))

        self.assertTrue(torch.allclose(
            results[0], golden_result, rtol=0.02, atol=0.02))


if __name__ == '__main__':
    unittest.main()
