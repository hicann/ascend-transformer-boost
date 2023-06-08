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
        operation = torch.classes.SelfAttentionLayerTorch.SelfAttentionLayerTorch()
        operation.test()
        input = torch.rand(384, 32, 1024).npu()
        qLinearWeight = torch.rand(1024, 1024).npu()
        qLinearBias = torch.rand(1024).npu()
        kLinearWeight = torch.rand(1024, 1024).npu()
        kLinearBias = torch.rand(1024).npu()
        vLinearWeight = torch.rand(1024, 1024).npu()
        vLinearBias = torch.rand(1024).npu()
        attentionMask = torch.rand(32, 1, 1, 384).npu()
        context = torch.rand(32, 1, 1, 384).npu()
        operation.execute(
            [input, qLinearWeight, qLinearBias, kLinearWeight, kLinearBias, vLinearWeight, vLinearBias, attentionMask], [context])


if __name__ == '__main__':
    unittest.main()
