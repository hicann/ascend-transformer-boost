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


OP_NAME = "SelfAttentionKvCacheFusionOperation"
PARAM = '{"headNum": 32, "layerId": 0, "dk": 128, "seqLen": [1], "tokenOffset": [22], \
        "numHeadsPerPartition": 32, "hiddenSizePerHead": 128, "numGroupsPerPartition": 2, "model": "chatglm2_6b"}'

INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor0.pth")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor1.pth")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor2.pth")
INTENSOR3 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor3.pth")
INTENSOR4 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor4.pth")
INTENSOR5 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor5.pth")
INTENSOR6 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor6.pth")
INTENSOR7 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor7.pth")
INTENSOR8 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "intensor8.pth")
OUTTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "tensors/operations/self_attention_kv_cache_fusion/chatglm2_6b", "outtensor0.pth")


class TestSelfAttentionKvCacheFusionGlm26b(operation_test.OperationTest):
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
                                      self.get_tensor(INTENSOR8).npu()])


if __name__ == '__main__':
    unittest.main()
