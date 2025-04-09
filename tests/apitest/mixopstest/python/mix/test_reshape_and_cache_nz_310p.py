# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
import os
import unittest
import numpy as np
import torch
import random
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import op_test

class TestReshapeAndCacheNz310p(op_test.OpTest):
    def setUp(self):
        super().setUp()

    def set_reshape_and_cache_nz_param(self, num_tokens, num_heads, head_size, block_size, num_blocks):
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = "float16"
        self.key = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
        self.value = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
        num_slots = self.block_size * self.num_blocks
        slot_list = random.sample(range(-num_slots, num_slots), self.num_tokens)
        self.slot_mapping = np.array(slot_list).astype(np.int32)

    def generate_test_data(self):
        key_expect_nz = np.zeros((self.num_blocks, self.num_heads * self.head_size // 16, self.block_size, 16)).astype(self.dtype)
        value_expect_nz = np.zeros((self.num_blocks, self.num_heads * self.head_size // 16, self.block_size, 16)).astype(self.dtype)

        for i, slot in enumerate(self.slot_mapping):
            if slot < 0:
                continue
            block_index = slot // self.block_size
            block_offset = slot % self.block_size

            token_key = self.key[i]
            token_v = self.value[i]
            token_key = token_key.reshape(self.num_heads * self.head_size)
            token_v = token_v.reshape(self.num_heads * self.head_size)
            for k in range(self.num_heads * self.head_size // 16):
                key_expect_nz[block_index][k][block_offset][:] = token_key[k * 16: k * 16 + 16]
                value_expect_nz[block_index][k][block_offset][:] = token_v[k * 16: k * 16 + 16]
        return key_expect_nz, value_expect_nz

    def golden_calc(self, in_tensors):
        tensor_out1, tensor_out2 = self.generate_test_data()
        logging.debug(f'kv_cache shape: , {tensor_out1.shape}, {tensor_out2.shape}')
        return [torch.tensor(tensor_out1), torch.tensor(tensor_out2)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        self.assertTrue(len(out_tensors) == len(golden_out_tensors))
        result = []
        for i in range(len(out_tensors)):
            actual_output = out_tensors[i]
            golden_output = golden_out_tensors[i]
            result.append(torch.allclose(actual_output.half(), golden_output.half(), rtol=0.001, atol=0.001))
        logging.debug(f"result is : {all(result)}")
        return all(result)
    
    def __run_reshape_and_cache_case(self, batch, seq_len, num_heads, head_size, block_size, num_blocks):
        num_tokens = batch * seq_len

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 1}
        self.set_reshape_and_cache_nz_param(num_tokens, num_heads, head_size, block_size, num_blocks)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nz, self.format_nz, self.format_nd])

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache_nz = np.zeros((num_blocks, num_heads * head_size // 16, block_size, 16)).astype(self.dtype)
        value_cache_nz = np.zeros((num_blocks, num_heads * head_size // 16, block_size, 16)).astype(self.dtype)

        self.execute([torch.tensor(key).half(), torch.tensor(value).half(), torch.tensor(key_cache_nz).half(),
                      torch.tensor(value_cache_nz).half(), torch.tensor(slot_mapping)], [2, 3])

    @op_test.only_310p
    def test_reshape_and_cache_nz_case0(self):
        batch = 2
        seq_len = 1
        num_heads = 32
        head_size = 64
        block_size = 128
        num_blocks = 64
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size, block_size, num_blocks)

    @op_test.only_310b
    def test_reshape_and_cache_nz_case1(self):
        batch = 1
        seq_len = 1
        num_heads = 4
        head_size = 128
        block_size = 128
        num_blocks = 3
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size, block_size, num_blocks)

    @unittest.skip("not for pipeline")
    def atest_reshape_and_cache_nz_big_batch(self):
        batch = 200
        seq_len = 2048
        num_heads = 1
        head_size = 16
        block_size = 16
        num_blocks = 200 * 2048
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size, block_size, num_blocks)


if __name__ == '__main__':
    unittest.main()