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

torch.npu.set_device(torch.device(f"npu:{0}"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


OP_NAME = "SelfAttentionKvCacheFusionOperation"
PARAM = '{"headNum": 32, "seqLen": [1], "dk": 128, "layerId": [0], "tokenOffset": [4]}'
INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor0.pth")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor1.pth")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor2.pth")
INTENSOR3 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor3.pth")
INTENSOR4 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor4.pth")
INTENSOR5 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor5.pth")
INTENSOR6 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor6.pth")
INTENSOR7 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor7.pth")
INTENSOR8 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache_fusion", "intensor8.pth")
OUTTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "tensors/operations/self_attention_kv_cache_fusion", "outtensor0.pth")


class TestSelfAttentionKvCacheFusionOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(OUTTENSOR0).npu()]

    def test(self):
        intensor0 = self.get_tensor(INTENSOR0).npu().npu_format_cast(2)
        intensor1 = self.get_tensor(INTENSOR1).npu().npu_format_cast(2)
        intensor2 = self.get_tensor(INTENSOR2).npu().npu_format_cast(2)
        intensor3 = self.get_tensor(INTENSOR3).npu().npu_format_cast(2)
        intensor4 = self.get_tensor(INTENSOR4).npu().npu_format_cast(2)
        intensor5 = self.get_tensor(INTENSOR5).npu().npu_format_cast(2)
        origin_size = intensor5.shape[0]
        max_tensor = torch.zeros(2048, 2048).npu().to(torch.half)
        max_tensor[:origin_size, :origin_size] = intensor5.to(torch.half)
        intensor6 = self.get_tensor(INTENSOR6).npu()
        intensor7 = self.get_tensor(INTENSOR7).npu()
        intensor8 = self.get_tensor(INTENSOR8).npu()
        print("max_tensor:" + str(max_tensor))
        self.execute(OP_NAME, PARAM, [intensor0,
                                      intensor1,
                                      intensor2,
                                      intensor3,
                                      intensor4,
                                      max_tensor,
                                      intensor6,
                                      intensor7,
                                      intensor8])


if __name__ == '__main__':
    unittest.main()
