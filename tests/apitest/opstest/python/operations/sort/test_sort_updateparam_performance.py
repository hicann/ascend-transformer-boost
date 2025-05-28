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
import json
import unittest
import torch
import torch_npu
import torch.nn as nn
import logging
import json
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test

OP_NAME = "SortOperation"


class TestSortOperation(operation_test.OperationTest):    

    def test_3d_float(self):
        if  operation_test.get_soc_version() == 'Ascend910A' or \
            operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend910A\Ascend310B")
            return True

        intensor = torch.randint(-65504, 65504, (8192, 8192, 8192)).float().npu().half()

        operation = torch.classes.OperationTorch.OperationTorch(OP_NAME)

        PARAM = {"num": [8192]}
        createOperation_start_time = time.perf_counter()
        operation.set_param(json.dumps(PARAM))
        createOperation_end_time = time.perf_counter()
        print(f"createOperation time: {createOperation_end_time - createOperation_start_time} second")

        execute_start_time = time.perf_counter()
        operation.execute([intensor])
        execute_end_time = time.perf_counter()
        print(f"execute time: {execute_end_time - execute_start_time} second")

        PARAM = {"num": [8191]}
        updateParam_start_time = time.perf_counter()
        operation.update_param(json.dumps(PARAM))
        updateParam_end_time = time.perf_counter()
        print(f"updateParam time: {updateParam_end_time - updateParam_start_time} second")

        PARAM = {"num": [8192]}
        updateParam_start_time = time.perf_counter()
        operation.update_param(json.dumps(PARAM))
        updateParam_end_time = time.perf_counter()
        print(f"updateParam time: {updateParam_end_time - updateParam_start_time} second")

        execute_start_time = time.perf_counter()
        operation.execute([intensor])
        execute_end_time = time.perf_counter()
        print(f"execute time: {execute_end_time - execute_start_time} second")

if __name__ == '__main__':
    unittest.main()
