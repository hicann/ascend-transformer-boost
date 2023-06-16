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


sys.path.append(os.path.dirname(__file__))
import operation_test  # NOQA: E402


OP_NAME = "AnyOperation"
PARAM = {
    "inTensors": [
        "aTensor",
        "bTensor",
        "cTensor",
    ],
    "outTensors": [
        "resultTensor"
    ],
    "internalTensors": [
        "add0OutTensor",
    ],
    "nodes": [
        {
            "opName": "BroadcastOperation",
            "specificParam": {
                "broadcastType": 1
            },
            "inTensors": [
                "aTensor",
                "bTensor"
            ],
            "outTensors": [
                "add0OutTensor"
            ]
        },
        {
            "opName": "BroadcastOperation",
            "specificParam": {
                "broadcastType": 1
            },
            "inTensors": [
                "add0OutTensor",
                "cTensor"
            ],
            "outTensors": [
                "resultTensor"
            ]
        }
    ]
}


class TestAnyOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [in_tensors[0] + in_tensors[1] + in_tensors[2]]

    def test_2d_half(self):
        self.execute_out(OP_NAME, PARAM,
                         [torch.rand(1024, 1024).npu().half(), torch.rand(1024, 1024).npu().half(),
                          torch.rand(1024, 1024).npu().half()],
                         [torch.rand(1024, 1024).npu().half()])


if __name__ == '__main__':
    unittest.main()
