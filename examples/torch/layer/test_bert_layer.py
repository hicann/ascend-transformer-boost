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
import time
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
#         operation = torch.classes.BertLayerTorch.BertLayerTorch('{"transKey":true,"dk":3,"headNum":16}')

#         hiddenStatesId = torch.rand(384,  32, 1024).npu()

#         qLinearWeightId = torch.rand(1024, 1024).npu()
#         qLinearBiasId = torch.rand(1024).npu()
#         kLinearWeightId = torch.rand(1024, 1024).npu()
#         kLinearBiasId = torch.rand(1024).npu()
#         vLinearWeightId = torch.rand(1024, 1024).npu()
#         vLinearBiasId = torch.rand(1024).npu()
#         selfOutLinearWeightId = torch.rand(1024, 1024).npu()
#         selfOutLinearBiasId = torch.rand(1024).npu()
#         selfOutNormWeightId = torch.rand(1024).npu()
#         selfOutNormBiasId = torch.rand(1024).npu()
#         ffnLinearWeightId = torch.rand(4096, 1024).npu()
#         ffnLinearBiasId = torch.rand(4096).npu()
#         bertOutLinearWeightId = torch.rand(1024, 4096).npu()
#         bertOutLinearBiasId = torch.rand(1024).npu()
#         bertOutNormWeightId = torch.rand(1024).npu()
#         bertOutNormBiasId = torch.rand(1024).npu()

#         attentionMaskId = torch.rand(32, 1, 1, 384).npu()

#         bertLayerOutId = torch.empty(384, 32, 1024).npu()

#         start_time = time.time()
#         for i in range(1):
#             operation.execute(
#                 [hiddenStatesId, qLinearWeightId, qLinearBiasId, kLinearWeightId, kLinearBiasId, vLinearWeightId, vLinearBiasId,
#                  selfOutLinearWeightId, selfOutLinearBiasId, selfOutNormWeightId, selfOutNormBiasId,
#                  ffnLinearWeightId, ffnLinearBiasId, bertOutLinearWeightId, bertOutLinearBiasId, bertOutNormWeightId, bertOutNormBiasId, attentionMaskId], [bertLayerOutId])
#         end_time = time.time()
#         print("use time:", (end_time - start_time))


class TestBert(unittest.TestCase):
    def test_2d(self):
        operation = torch.classes.LayerTorch.LayerTorch("BertLayer")
        operation.set_param('{"transKey":true,"dk":64,"headNum":16}')
        testcase = TensorTestCase('BertLayer', in_tensor_num=18)
        testcase.read(1)
        in_tensors = testcase.get_in_tensors()
        out_tensors = testcase.get_out_tensors()
        hiddenStatesId = in_tensors[0].npu()

        qLinearWeightId = in_tensors[1].npu()
        qLinearBiasId = in_tensors[2].npu()
        kLinearWeightId = in_tensors[3].npu()
        kLinearBiasId = in_tensors[4].npu()
        vLinearWeightId = in_tensors[5].npu()
        vLinearBiasId = in_tensors[6].npu()
        selfOutLinearWeightId = in_tensors[7].npu()
        selfOutLinearBiasId = in_tensors[8].npu()
        selfOutNormWeightId = in_tensors[9].npu()
        selfOutNormBiasId = in_tensors[10].npu()
        ffnLinearWeightId = in_tensors[11].npu()
        ffnLinearBiasId = in_tensors[12].npu()
        bertOutLinearWeightId = in_tensors[13].npu()
        bertOutLinearBiasId = in_tensors[14].npu()
        bertOutNormWeightId = in_tensors[15].npu()
        bertOutNormBiasId = in_tensors[16].npu()

        attentionMaskId = in_tensors[17].npu()

        bertLayerOutId = torch.empty(384, 32, 1024).half().npu()
        operation.execute(
            [hiddenStatesId, qLinearWeightId, qLinearBiasId, kLinearWeightId, kLinearBiasId, vLinearWeightId, vLinearBiasId,
             selfOutLinearWeightId, selfOutLinearBiasId, selfOutNormWeightId, selfOutNormBiasId,
             ffnLinearWeightId, ffnLinearBiasId, bertOutLinearWeightId, bertOutLinearBiasId, bertOutNormWeightId, bertOutNormBiasId, attentionMaskId], [bertLayerOutId])

        ground_bertLayerOutId = out_tensors[0].npu()
        print("bertLayerOutId:" + str(bertLayerOutId.size()))
        print("ground_bertLayerOutId:" + str(ground_bertLayerOutId.size()))
        print("bertLayerOutId:" + str(bertLayerOutId))
        print("ground_bertLayerOutId:" + str(ground_bertLayerOutId))

        self.assertTrue(torch.allclose(
            bertLayerOutId, ground_bertLayerOutId, rtol=0.05, atol=0.05))


if __name__ == '__main__':
    unittest.main()
