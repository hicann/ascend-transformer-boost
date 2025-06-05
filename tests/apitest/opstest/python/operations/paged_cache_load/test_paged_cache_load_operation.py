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

MAX_SEQ_LEN = 1024

not_support_device = ['Ascend910A','Ascend310B','Ascend310P']

def generate_data(
        batch=4,
        seq_len=1,
        num_heads=2,
        head_size_k=32,
        head_size_v=32,
        block_size=128,
        num_blocks=4,
):
    soc_version = operation_test.get_soc_version()
    dtype = "float16"
    num_tokens = batch * seq_len
    key_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(
        np.float16)
    value_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_v // 16, block_size, 16)).astype(
        np.float16)

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
    max_context_len = max(context_lens)

    max_num_blocks_per_req = (max_context_len + block_size - 1) // block_size
    block_tables = []  # [num_tokens, max_num_blocks_per_seq]
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
        ]
        block_tables.append(block_table)

    context_lens = np.array(context_lens).astype(np.int32)
    block_tables = np.array(block_tables).astype(np.int32)
    sum_context_lens = sum(context_lens)
    key_expect = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
    value_expect = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)
    key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
    value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)

    elenum_aligned = 16
    kv_rslt_id = 0

    for i in range(num_tokens):
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        for j in range(context_len):
            block_id = int(block_table[j // block_size])
            block_offset = j % block_size

            if block_id < 0:
                continue

            temp_k = np.zeros((num_heads * head_size_k))
            temp_v = np.zeros((num_heads * head_size_v))

            for k in range(num_heads * head_size_k // elenum_aligned):
                temp_k[k * elenum_aligned: k * elenum_aligned + elenum_aligned] = key_cache[block_id][k][block_offset][
                                                                                  :]

            for k in range(num_heads * head_size_v // elenum_aligned):
                temp_v[k * elenum_aligned: k * elenum_aligned + elenum_aligned] = value_cache[block_id][k][
                                                                                      block_offset][:]

            key_expect[kv_rslt_id] = temp_k
            value_expect[kv_rslt_id] = temp_v
            kv_rslt_id += 1

    ret_data = key_cache, value_cache, block_tables,context_lens, key_expect, value_expect, key, value
    return ret_data


OP_NAME = "PagedCacheLoadOperation"
PARAM = json.dumps({"type": 1})

data = generate_data()
in_tensors = [torch.from_numpy(tensor) for tensor in data]
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]


class ReshapeAndCacheGradOperation(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()

    def golden_calc(self, input_tensors):
        return [in_tensors[4], in_tensors[5]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.equal(out_tensor, golden_out_tensor)

    def test(self):
        # 获取Soc型号
        if operation_test.get_soc_version() in not_support_device:
            print("These test cases only support A2/A3")
            return True
        in_tensors[0] = torch_npu.npu_format_cast(in_tensors[0], 29)
        in_tensors[1] = torch_npu.npu_format_cast(in_tensors[1], 29)
        self.execute_out(OP_NAME, PARAM, [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[6], in_tensors[7]], [in_tensors[4], in_tensors[5]])
        #RUN_PARAM = json.dumps({"contextLens": in_tensors[3].cpu().numpy().tolist(),"type": 1})
        #self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3]])

if __name__ == '__main__':
    unittest.main()