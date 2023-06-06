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
import unittest
import os
import json
import torch
import torch_npu


ACLTRANSFORMER_HOME_PATH = os.environ.get("ACLTRANSFORMER_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")

LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH,
                        "examples/libacltransformer_torch.so")
torch.classes.load_library(LIB_PATH)


class TestAddNormal(unittest.TestCase):
    # [seq, batch, headNum, headSize] = [1, 1, 32, 128]
    def test_2d(self):
        operation = torch.classes.OperationTorch.OperationTorch()
        operation.test()
        # in
        query = torch.rand(4, 1, 32, 128).npu()
        key = torch.rand(4, 1, 32, 128).npu()
        value = torch.rand(4, 1, 32, 128).npu()
        attention_mask = torch.ones(1, 1, 4, 15, dtype=torch.bool).npu()
        past_key = torch.rand(11, 1, 32, 128).npu()
        past_value = torch.rand(11, 1, 32, 128).npu()

        result, present_key, present_value = operation.execute("SelfAttentionKvCacheOperation",
                          json.dumps({"transKey": True, "dk": 128,
                                     "headNum": 32, "layerId": 13}),
                          [query, key, value, attention_mask, past_key, past_value])
        print("result:" + str(result.shape))
        print("present_key:" + str(present_key.shape))
        print("present_value:" + str(present_value.shape))


if __name__ == '__main__':
    unittest.main()
