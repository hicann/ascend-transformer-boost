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
import operation_test

OP_NAME = "PositionEmbeddingOperation"
PARAM = '{"is2d": false, "headNum": 12, "numHeadsPerPartition": 0, "hiddenSizePerHead": 0, "numGroupsPerPartition": 0, "dk": 0, "rotaryPct": 0, "model": "", "isFusion": true}'

INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "/data/acltransformer_testdata/tensors/operations/position_embedding_1d_mixed_fusion/",
                         "inTensor0.bin")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "/data/acltransformer_testdata/tensors/operations/position_embedding_1d_mixed_fusion/",
                         "inTensor1.bin")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "/data/acltransformer_testdata/tensors/operations/position_embedding_1d_mixed_fusion/",
                         "inTensor2.bin")
INTENSOR3 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "/data/acltransformer_testdata/tensors/operations/position_embedding_1d_mixed_fusion/",
                         "inTensor3.bin")

OUTTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "/data/acltransformer_testdata/tensors/operations/position_embedding_1d_mixed_fusion/",
                          "outTensor0.bin")
OUTTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "/data/acltransformer_testdata/tensors/operations/position_embedding_1d_mixed_fusion/",
                          "outTensor1.bin")
OUTTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "/data/acltransformer_testdata/tensors/operations/position_embedding_1d_mixed_fusion/",
                          "outTensor2.bin")


class TestPositionEmbedding1dMixedFusion(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(OUTTENSOR0).npu(),
                self.get_tensor(OUTTENSOR1).npu(),
                self.get_tensor(OUTTENSOR2).npu()]

    def test(self):
        self.execute(OP_NAME, PARAM, [self.get_tensor(INTENSOR0).npu(),
                                      self.get_tensor(INTENSOR1).npu(),
                                      self.get_tensor(INTENSOR2).npu(),
                                      self.get_tensor(INTENSOR3).npu()])


if __name__ == '__main__':
    unittest.main()
