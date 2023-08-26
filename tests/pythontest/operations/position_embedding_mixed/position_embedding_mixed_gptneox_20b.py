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


OP_NAME = "PositionEmbeddingOperation"
PARAM = {"dk": 96, "rotaryPct": 0.25, "headNum": 64, "model": "gptneox20b"}

INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_mixed/gptneox_20b", "inTensor0.bin")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_mixed/gptneox_20b", "inTensor1.bin")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_mixed/gptneox_20b", "inTensor2.bin")
INTENSOR3 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_mixed/gptneox_20b", "inTensor3.bin")

OUTTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_mixed/gptneox_20b", "outTensor0.bin")
OUTTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_mixed/gptneox_20b", "outTensor1.bin")
OUTTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/position_embedding_mixed/gptneox_20b", "outTensor2.bin")

class TestPositionEmbeddingMixedGptneox20B(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(OUTTENSOR0).npu(),
                self.get_tensor(OUTTENSOR1).npu(),
                self.get_tensor(OUTTENSOR2).npu()]

    def test(self):
        mixed_qkv = self.get_tensor(INTENSOR0).npu()
        position_ids = self.get_tensor(INTENSOR1).npu()
        cos_table = self.get_tensor(INTENSOR2).npu()
        sin_table = self.get_tensor(INTENSOR3).npu()
        acl_cos_embed = torch.nn.functional.embedding(position_ids, cos_table).half()
        acl_sin_embed = torch.nn.functional.embedding(position_ids, sin_table).half()
        self.execute(OP_NAME, PARAM, [mixed_qkv, position_ids, acl_cos_embed, acl_sin_embed])


if __name__ == '__main__':
    unittest.main()
