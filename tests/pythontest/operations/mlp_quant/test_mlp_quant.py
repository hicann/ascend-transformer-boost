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
import numpy as np


OP_NAME = "MlpQuantOperation"
op_param = np.load(os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                                "tensors/operations/mlp_quant", "test_mlp_quant_param.npy"))

PARAM = {"model": "llama7b", "inputScale": float(op_param[0]), "inputOffset": int(op_param[1])}

INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_hidden_states.pth")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_gate_weight.pth")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_gate_deq_scale.pth")
INTENSOR3 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_gate_bias.pth")
INTENSOR4 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_down_weight.pth")
INTENSOR5 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_down_deq_scale.pth")
INTENSOR6 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_down_bias.pth")
INTENSOR7 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_up_weight.pth")
INTENSOR8 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_up_deq_scale.pth")
INTENSOR9 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/mlp_quant", "test_up_bias.pth")
OUTTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "tensors/operations/mlp_quant", "test_golden.pth")


class TestMlpQuantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(OUTTENSOR0).npu()]

    def test(self):
        self.execute(OP_NAME, PARAM, [self.get_tensor(INTENSOR0).npu(),
                                      self.get_tensor(INTENSOR1).npu(),
                                      self.get_tensor(INTENSOR2).npu(),
                                      self.get_tensor(INTENSOR3).npu(),
                                      self.get_tensor(INTENSOR4).npu(),
                                      self.get_tensor(INTENSOR5).npu(),
                                      self.get_tensor(INTENSOR6).npu(),
                                      self.get_tensor(INTENSOR7).npu(),
                                      self.get_tensor(INTENSOR8).npu(),
                                      self.get_tensor(INTENSOR9).npu()])


if __name__ == '__main__':
    unittest.main()
