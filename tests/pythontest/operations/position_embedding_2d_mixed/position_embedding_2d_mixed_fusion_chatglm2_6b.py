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


OP_NAME = "RopeOperation"
PARAM = {"headNum": 32}
INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "/data/acltransformer_testdata/tensors/operations/position_embedding_2d_mixed_fusion/chatglm2_6b", "inTensor0.bin")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "/data/acltransformer_testdata/tensors/operations/position_embedding_2d_mixed_fusion/chatglm2_6b", "inTensor1.bin")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "/data/acltransformer_testdata/tensors/operations/position_embedding_2d_mixed_fusion/chatglm2_6b", "inTensor2.bin")

OUTTENSOR0 = os.path.join(os.getenv(
    "ACLTRANSFORMER_TESTDATA"), "/data/acltransformer_testdata/tensors/operations/position_embedding_2d_mixed_fusion/chatglm2_6b",
                          "outTensor0.bin")
OUTTENSOR1 = os.path.join(os.getenv(
    "ACLTRANSFORMER_TESTDATA"), "/data/acltransformer_testdata/tensors/operations/position_embedding_2d_mixed_fusion/chatglm2_6b",
                          "outTensor1.bin")
OUTTENSOR2 = os.path.join(os.getenv(
    "ACLTRANSFORMER_TESTDATA"), "/data/acltransformer_testdata/tensors/operations/position_embedding_2d_mixed_fusion/chatglm2_6b",
                          "outTensor2.bin")

class RopeOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(OUTTENSOR0).npu(),
                self.get_tensor(OUTTENSOR1).npu(),
                self.get_tensor(OUTTENSOR2).npu()]

    def test(self):
        self.execute(OP_NAME, PARAM, [self.get_tensor(INTENSOR0).npu(),
                                      self.get_tensor(INTENSOR1).npu(),
                                      self.get_tensor(INTENSOR2).npu()])


if __name__ == '__main__':
    unittest.main()
