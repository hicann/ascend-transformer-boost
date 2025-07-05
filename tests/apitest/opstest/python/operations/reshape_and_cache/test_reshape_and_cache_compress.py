#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import json
import math
import os
import random
import sys
import unittest
 
import numpy as np
import torch
import torch_npu
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
from reshape_and_cache.reshape_and_cache_data_generator import ReshapeAndCacheDataGenerator
 
MAX_SEQ_LEN = 1024
 
 

 
OP_NAME = "ReshapeAndCacheOperation"
PARAM = json.dumps({"compressType": 1})
 
data_generator = ReshapeAndCacheDataGenerator()
data_generator.test_reshape_and_cache_compress_head()
data = data_generator.generate_test_data()
if data_generator.dtype == "bfloat16":
    in_tensors = []
    for tensor in data:
        pt_tensor = torch.from_numpy(tensor)
        if pt_tensor.is_floating_point():
            pt_tensor = pt_tensor.to(torch.bfloat16)
        in_tensors.append(pt_tensor)
else:
    in_tensors = [torch.from_numpy(tensor) for tensor in data]
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

 
class TestReshapeAndCacheOperationCompress(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors[7], in_tensors[8]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        if data_generator.dtype == "bfloat16":
            out = out_tensor.cpu()
            golden = golden_out_tensor.cpu()
            out = out.to(torch.float32)
            golden = golden.to(torch.float32)

            return torch.equal(out, golden)
        else:
            return torch.equal(out_tensor.cpu(), golden_out_tensor.cpu())
 
    def test(self):
        if not TestReshapeAndCacheOperationCompress.soc_version in ['Ascend910B', 'Ascend910_9599']:
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM, [in_tensors[0], in_tensors[1], in_tensors[2],\
                in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6]],
                [in_tensors[2], in_tensors[3]])
 
 
if __name__ == '__main__':
    unittest.main()