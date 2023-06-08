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
import time
import torch
import torch_npu
import sys


ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "examples/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)


class TestNormal(unittest.TestCase):
    def test_2d(self):
        # operation = torch.classes.GlmBlock.GlmBlock(
        #     json.dumps({"transKey": True, "dk": 128, "headNum": 32, "layerId": 0,
        #                 "layerNormEps":1e-12, "ResidualAddScale": 0}))

        # hiddenStates = torch.rand(384,  32, 1024).npu()
        # normWeight = torch.rand(4096).npu()
        # normBias = torch.rand(4096).npu()
        # qkvMixdWeight = torch.rand(12288, 4096).npu()
        # qkvMixdBias = torch.rand(12288).npu()
        # selfOutLinearWeight = torch.rand(4096, 4096).npu()
        # selfOutLinearBias = torch.rand(4096).npu()
        # selfOutNormWeight = torch.rand(4096).npu()
        # selfOutNormBias = torch.rand(4096).npu()
        # ffnLinearWeight = torch.rand(16384, 4096).npu()
        # ffnLinearBias = torch.rand(16384).npu()
        # ffnOutLinearWeight = torch.rand(4096, 16384).npu()
        # ffnOutLinearBias = torch.rand(4096).npu()
        # positionIds = torch.rand(384,  32, 1024).npu()
        # cosTable = torch.rand(2049, 1, 1024).npu()
        # sinTable = torch.rand(2049, 1, 1024).npu()
        # attentionMask = torch.rand(384,  32, 1024).npu()
        # pastKey = torch.rand(384,  32, 1024).npu()
        # pastValue = torch.rand(384,  32, 1024).npu()



        # start_time = time.time()
        # for i in range(1):
        #     operation.execute(
        #         [hiddenStates, normWeight, normBias, qkvMixdWeight, qkvMixdBias, selfOutLinearWeight,
        #          selfOutLinearBias, selfOutNormWeight, selfOutNormBias, ffnLinearWeight,
        #          ffnLinearBias, ffnOutLinearWeight, ffnOutLinearBias, positionIds, cosTable, sinTable, 
        #          attentionMask, pastKey, pastValue], [])
        # end_time = time.time()
        # print("use time:", (end_time - start_time))


if __name__ == '__main__':
    unittest.main()
