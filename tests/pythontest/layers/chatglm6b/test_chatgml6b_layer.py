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
import math
import torch
import torch_npu


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import layer_test  # NOQA: E402


LAYER_NAME = "ChatGlm6BLayer"
PARAM = {"transKey": True, "dk": 128, "headNum": 32, "layerId": 0,
         "layerNormEps": 1e-5, "ResidualAddScale": math.sqrt(2 * 28)}
INTENSOR0 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor0.pth")
INTENSOR1 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor1.pth")
INTENSOR2 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor2.pth")
INTENSOR3 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor3.pth")
INTENSOR4 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor4.pth")
INTENSOR5 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor5.pth")
INTENSOR6 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor6.pth")
INTENSOR7 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor7.pth")
INTENSOR8 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor8.pth")
INTENSOR9 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor9.pth")
INTENSOR10 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor10.pth")
INTENSOR11 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor11.pth")
INTENSOR12 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor12.pth")
INTENSOR13 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor13.pth")
INTENSOR14 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor14.pth")
INTENSOR15 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor15.pth")
INTENSOR16 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor16.pth")
INTENSOR17 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor17.pth")
INTENSOR18 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "intensor18.pth")
OUTTENSOR0 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "outtensor0.pth")
OUTTENSOR1 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "outtensor1.pth")
OUTTENSOR2 = os.path.join(
    os.getenv("ACLTRANSFORMER_TESTDATA"), "tensors/layers/chatglm6b", "outtensor2.pth")


class TestChatGlm6BLayer(layer_test.LayerTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(OUTTENSOR0).npu(),
                self.get_tensor(OUTTENSOR1).npu(),
                self.get_tensor(OUTTENSOR2).npu()]

    def test_2d_float(self):
        self.execute(LAYER_NAME, PARAM, [
            self.get_tensor(INTENSOR0).npu(),
            self.get_tensor(INTENSOR1).npu(),
            self.get_tensor(INTENSOR2).npu(),
            self.get_tensor(INTENSOR3).npu(),
            self.get_tensor(INTENSOR4).npu(),
            self.get_tensor(INTENSOR5).npu(),
            self.get_tensor(INTENSOR6).npu(),
            self.get_tensor(INTENSOR7).npu(),
            self.get_tensor(INTENSOR8).npu(),
            self.get_tensor(INTENSOR9).npu(),
            self.get_tensor(INTENSOR10).npu(),
            self.get_tensor(INTENSOR11).npu(),
            self.get_tensor(INTENSOR12).npu(),
            self.get_tensor(INTENSOR13).npu(),
            self.get_tensor(INTENSOR14).npu(),
            self.get_tensor(INTENSOR15).npu(),
            self.get_tensor(INTENSOR16).npu(),
            self.get_tensor(INTENSOR17).npu(),
            self.get_tensor(INTENSOR18).npu(),
        ])


if __name__ == '__main__':
    unittest.main()
