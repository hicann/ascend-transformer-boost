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


OP_NAME = "RmsNormOperation"
PARAM = {"rmsNormEps": 1e-6}
INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/rms_norm", "intensor0.pth")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/rms_norm", "intensor1.pth")
OUTTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "tensors/operations/rms_norm", "outtensor0.pth")

class TestSelfAttentionKvCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        print("golden output tensor")
        print(self.get_tensor(OUTTENSOR0))
        return [self.get_tensor(OUTTENSOR0).npu()]

    def test(self):
        self.execute(OP_NAME, PARAM, [self.get_tensor(INTENSOR0).npu(),
                                      self.get_tensor(INTENSOR1).npu()])


if __name__ == '__main__':
    unittest.main()
