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
PARAM = '{"headNum": 32, "layerId": 0, "dk": 128, "seqLen": [1], "tokenOffset": [5], "model": "chatglm6b"}'

INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor0.bin")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor1.bin")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor2.bin")
INTENSOR3 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor3.bin")
INTENSOR4 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor4.bin")
INTENSOR5 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor5.bin")
INTENSOR6 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor6.bin")
INTENSOR7 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor7.bin")
INTENSOR8 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "inTensor8.bin")
OUTTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "tensors/operations/self_attention_kv_cache_fusion/chatglm6b", "outTensor0.bin")


class TestSelfAttentionKvCacheFusionGlm6bOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(OUTTENSOR0).npu()]

    def test(self):
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version in [104, 220, 221, 222, 223]:
            self.execute(OP_NAME, PARAM, [self.get_tensor(INTENSOR0).npu(),
                                          self.get_tensor(INTENSOR1).npu(),
                                          self.get_tensor(INTENSOR2).npu(),
                                          self.get_tensor(INTENSOR3).npu(),
                                          self.get_tensor(INTENSOR4).npu(),
                                          self.get_tensor(INTENSOR5).npu(),
                                          self.get_tensor(INTENSOR6).npu(),
                                          self.get_tensor(INTENSOR7).npu(),
                                          self.get_tensor(INTENSOR8).npu()])
        else:
            print("310p skip")


if __name__ == '__main__':
    unittest.main()
