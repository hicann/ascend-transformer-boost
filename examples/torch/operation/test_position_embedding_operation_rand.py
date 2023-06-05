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
        operation = torch.classes.OperationTorch.OperationTorch(
            "PositionEmbeddingOperation")
        operation.set_param(json.dumps({"headNum": 32}))
        # in
        query = torch.rand(1, 1, 4096).npu()
        position_ids = torch.rand(1, 2, 4).npu()
        cos_table = torch.rand(1, 1, 4).npu()
        sin_table = torch.rand(1, 1, 4).npu()

        result = operation.execute([query, position_ids, cos_table, sin_table])
        print("result:" + str(result.shape))


if __name__ == '__main__':
    unittest.main()
