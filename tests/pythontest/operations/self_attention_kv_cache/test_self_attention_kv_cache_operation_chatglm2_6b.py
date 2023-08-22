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
ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

TOOLS_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH, "tools/")

sys.path.append(TOOLS_PATH)

import operation_test  # NOQA: E402
import tensor_file


OP_NAME = "SelfAttentionKvCacheOperation"
PARAM = '{"model": "chatglm2_6b", "headNum": 32, "layerId": 0, "dk": 128, "numGroupsPerPartition": 2, "numHeadsPerPartition": 32, "hiddenSizePerHead": 128, "postScale": 1, "preScale": 1}'
INTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache/chatglm2_6b", "inTensor0.bin")
INTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache/chatglm2_6b", "inTensor1.bin")
INTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache/chatglm2_6b", "inTensor2.bin")
INTENSOR3 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache/chatglm2_6b", "inTensor3.bin")
INTENSOR4 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                         "tensors/operations/self_attention_kv_cache/chatglm2_6b", "inTensor4.bin")
OUTTENSOR0 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "tensors/operations/self_attention_kv_cache/chatglm2_6b", "outTensor0.bin")
OUTTENSOR1 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "tensors/operations/self_attention_kv_cache/chatglm2_6b", "outTensor1.bin")
OUTTENSOR2 = os.path.join(os.getenv("ACLTRANSFORMER_TESTDATA"),
                          "tensors/operations/self_attention_kv_cache/chatglm2_6b", "outTensor2.bin")


class TestSelfAttentionKvCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [tensor_file.read_tensor(OUTTENSOR0).npu(),
                tensor_file.read_tensor(OUTTENSOR1).npu(),
                tensor_file.read_tensor(OUTTENSOR2).npu()]

    def test(self):
        self.execute(OP_NAME, PARAM, [tensor_file.read_tensor(INTENSOR0).npu(),
                                      tensor_file.read_tensor(INTENSOR1).npu(),
                                      tensor_file.read_tensor(INTENSOR2).npu(),
                                      tensor_file.read_tensor(INTENSOR3).npu(),
                                      tensor_file.read_tensor(INTENSOR4).npu()])


if __name__ == '__main__':
    unittest.main()
