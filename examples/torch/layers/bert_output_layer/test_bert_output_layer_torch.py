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
        operation = torch.classes.BertOutputLayerTorch.BertOutputLayerTorch()
        input = torch.rand(2, 3).npu()
        linearWeight = torch.rand(3, 2).npu()
        linearBias = torch.rand(2, 2).npu()
        residualAddIn = torch.rand(2, 2).npu()
        layerNormWeight = torch.rand(2).npu()
        layerNormBias = torch.rand(2).npu()
        bertOut = torch.empty(2, 3).npu()
        print("input:" + str(input))
        print("linearWeight:" + str(linearWeight))
        print("linearBias:" + str(linearBias))
        print("residualAddIn:" + str(residualAddIn))
        print("layerNormWeight:" + str(layerNormWeight))
        print("layerNormBias:" + str(layerNormBias))
        operation.execute(
            [input, linearWeight, linearBias, residualAddIn, layerNormWeight, layerNormBias], [bertOut])

        linear = torch.nn.Linear(3, 2)
        linear.weight.data = linearWeight
        linear.bias.data = linearBias
        layerNorm = torch.nn.norm(2, 1e-12)
        layerNorm.weight.data = layerNormWeight
        layerNorm.bias.data = layerNormBias

        golden_bertOut = layerNorm(linear(input) + input)
        print("c:" + str(bertOut))
        print("golden_c:" + str(golden_bertOut))

        self.assertTrue(torch.allclose(
            bertOut, golden_bertOut, rtol=0.02, atol=0.02))


if __name__ == '__main__':
    unittest.main()
