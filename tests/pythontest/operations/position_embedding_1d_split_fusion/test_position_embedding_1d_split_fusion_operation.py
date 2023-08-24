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
import sys
import os
import unittest
import torch
import torch_npu

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import operation_test  # NOQA: E402


OP_NAME = "PositionEmbedding1dSplitFusionOperation"
PARAM = {"headNum": 32, "rmsNormEps": 1e-6, "dk": 128, "model": "llama7b", "rotaryCoeff" :2}

INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_1d_split_fusion/after", "inTensor0.bin")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_1d_split_fusion/after", "inTensor1.bin")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_1d_split_fusion/after", "inTensor2.bin")
INTENSOR3 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_1d_split_fusion/after", "inTensor3.bin")
INTENSOR4 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_1d_split_fusion/after", "inTensor4.bin")
INTENSOR5 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_1d_split_fusion/after", "inTensor5.bin")
OUTTENSOR0 = os.path.join(os.getenv(
    "ACLTRANSFORMER_TESTDATA"), "tensors/operations/position_embedding_1d_split_fusion/after", "outTensor0.bin")
OUTTENSOR1 = os.path.join(os.getenv(
    "ACLTRANSFORMER_TESTDATA"), "tensors/operations/position_embedding_1d_split_fusion/after", "outTensor1.bin")


class TestPositionEmbeddingOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(OUTTENSOR0).npu(),
                self.get_tensor(OUTTENSOR1).npu()]
                
    def test(self):
        self.execute(OP_NAME, PARAM, [self.get_tensor(INTENSOR0).npu(),
                                      self.get_tensor(INTENSOR1).npu(),
                                      self.get_tensor(INTENSOR2).npu(),
                                      self.get_tensor(INTENSOR3).npu(),
                                      self.get_tensor(INTENSOR4).npu(),
                                      self.get_tensor(INTENSOR5).npu()
                                      ])


if __name__ == '__main__':
    os.environ['ACLTRANSFORMER_CONVERT_NCHW_TO_ND'] = '1'
    unittest.main()
    os.environ['ACLTRANSFORMER_CONVERT_NCHW_TO_ND'] = '0'
