#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import torch
import torch_npu
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "FillOperation"
PARAM = {"withMask": True, "value": -10000}


class TestFillOperation(operation_test.OperationTest):

    def test_float16(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        intensor0 = torch.rand(10000, 10000).npu().half()
        intentor1 = (torch.randint(2, (10000, 10000)) ==
                     torch.randint(1, (10000, 10000))).to(torch.bool).npu()

        operation = torch.classes.OperationTorch.OperationTorch(OP_NAME)

        PARAM = {"withMask": True, "value": -10000}
        createOperation_start_time = time.perf_counter()
        operation.set_param(json.dumps(PARAM))
        createOperation_end_time = time.perf_counter()
        print(f"createOperation time: {createOperation_end_time - createOperation_start_time} second")

        execute_1_start_time = time.perf_counter()
        operation.execute([intensor0, intentor1])
        execute_1_end_time = time.perf_counter()
        print(f"execute_1 time: {execute_1_end_time - execute_1_start_time} second")

        PARAM = {"withMask": True, "value": 10000}
        updateParam_start_time = time.perf_counter()
        operation.update_param(json.dumps(PARAM))
        updateParam_end_time = time.perf_counter()
        print(f"updateParam time: {updateParam_end_time - updateParam_start_time} second")

        execute_2_start_time = time.perf_counter()
        operation.execute([intensor0, intentor1])
        execute_2_end_time = time.perf_counter()
        print(f"execute_2 time: {execute_2_end_time - execute_2_start_time} second")


if __name__ == '__main__':
    unittest.main()