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
import sys
import json
import time
import torch
import torch_npu


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import layer_test  # NOQA: E402


LAYER_NAME = "BertLayer"


class TestNormal(layer_test.LayerTest):
    def golden_calc(self, in_tensors):
        return [in_tensors[0]]

    def test_2d_float(self):
        hiddenStatesId = torch.rand(384,  32, 1024).npu()
        qLinearWeightId = torch.rand(1024, 1024).npu()
        qLinearBiasId = torch.rand(1024).npu()
        kLinearWeightId = torch.rand(1024, 1024).npu()
        kLinearBiasId = torch.rand(1024).npu()
        vLinearWeightId = torch.rand(1024, 1024).npu()
        vLinearBiasId = torch.rand(1024).npu()
        selfOutLinearWeightId = torch.rand(1024, 1024).npu()
        selfOutLinearBiasId = torch.rand(1024).npu()
        selfOutNormWeightId = torch.rand(1024).npu()
        selfOutNormBiasId = torch.rand(1024).npu()
        ffnLinearWeightId = torch.rand(4096, 1024).npu()
        ffnLinearBiasId = torch.rand(4096).npu()
        bertOutLinearWeightId = torch.rand(1024, 4096).npu()
        bertOutLinearBiasId = torch.rand(1024).npu()
        bertOutNormWeightId = torch.rand(1024).npu()
        bertOutNormBiasId = torch.rand(1024).npu()
        attentionMaskId = torch.rand(32, 1, 1, 384).npu()
        bertLayerOutId = torch.empty(384, 32, 1024).npu()

        self.execute(LAYER_NAME, '{"transKey":true,"dk":3,"headNum":16}', [hiddenStatesId, qLinearWeightId, qLinearBiasId, kLinearWeightId, kLinearBiasId, vLinearWeightId, vLinearBiasId,
                                                                           selfOutLinearWeightId, selfOutLinearBiasId, selfOutNormWeightId, selfOutNormBiasId,
                                                                           ffnLinearWeightId, ffnLinearBiasId, bertOutLinearWeightId, bertOutLinearBiasId, bertOutNormWeightId, bertOutNormBiasId, attentionMaskId], [bertLayerOutId])


if __name__ == '__main__':
    unittest.main()
