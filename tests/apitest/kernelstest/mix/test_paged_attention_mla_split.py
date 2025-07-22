# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import logging
import sys
import os
import unittest
import math
import numpy as np

sys.path.append('../')
sys.path.append('../..')
import op_test
import torch
import random
import sys
import numpy as np
import math

np.random.seed(1)
from precision_calcu import *


class TestPagedAttentionMLA(op_test.OpTest):

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.flatten().to(torch.float32)
        out = out.flatten().to(torch.float32)
        len = out.shape[0]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        logging.info(f"maxDiff {max_diff}")
        logging.info("1/1000 Accuracy is %f", 1 - float(error_count) / len)
        logging.info("5/1000 Accuracy is %f", 1 - float(strict_error_count) / len)
        if self.data_type == torch.bfloat16 or self.is_int8_flag:
            logging.info("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[2])
        else:
            logging.info("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[0])
        calc_times = self.head_size_qk * self.max_context_len + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2 ** (-7)
            else:
                error = 2 ** (-6)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res
        else:
            if calc_times < 2048:
                error = 2 ** (-8)
            else:
                error = 2 ** (-7)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res

    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        # slopes = torch.ones(n_heads)
        return slopes

    def group_mm_torch(self, heads, group_num, A, B, is_k):
        group_head = heads // group_num
        score_high = None
        for i in range(group_num):
            if self.is_int8_flag:
                int8_B = B[i: (i + 1), :, :, ]
                head_dim = int8_B.shape[2]
                int32_B = torch.matmul(torch.eye(int8_B.shape[1]).to(torch.float32), int8_B.to(torch.float32)).to(
                    torch.int32)
                if is_k:
                    if self.has_bias:
                        int32_B = int32_B + self.offset1[i * head_dim:(i + 1) * head_dim]
                    fp32_B = int32_B.to(torch.float32) * self.de_scale1_fp32[i * head_dim:(i + 1) * head_dim]
                    fp32_B = torch.permute(fp32_B, (0, 2, 1))
                else:
                    if self.has_bias:
                        int32_B = int32_B + self.offset2[i * head_dim:(i + 1) * head_dim]
                    fp32_B = int32_B.to(torch.float32) * self.de_scale2_fp32[i * head_dim:(i + 1) * head_dim]
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                                fp32_B)
            else:
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                                B[i:(i + 1), :, :].to(torch.float32))
            if score_high is None:
                score_high = group_score_high
            else:
                score_high = torch.cat((score_high, group_score_high), 0)
        return score_high

    def process_deq_scale(self, deq_scale) -> np.ndarray:
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
        return new_deq_scale.astype(np.uint64)

    def softmax(self, sim):
        row_max = torch.max(sim, axis=-1, keepdims=True)[0]
        sim_sub = sim - row_max
        sim_sub = torch.exp(sim_sub)
        row_sum = torch.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def softmax_numpy(self, sim):
        sim = sim.cpu().numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        # print(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res, row_max + np.log(row_sum)

    def shape_nd_to_nz(self, shape, dtype='float16'):
        assert len(shape) >= 2
        batch = shape[:-2]  # 最后两维nd->nz
        a, b = shape[-2], shape[-1]
        a0, b0 = 16, 16
        return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

    def gen_axes_for_transpose(self,offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def convert_nd_to_nz(self, x):
        array_trans = self.gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3])  # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
        x_shape = self.shape_nd_to_nz(x.shape, dtype=x.dtype)
        *_, n1, m1, m0, n0 = x_shape
        return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).permute(*array_trans)  # x原始需要对齐，才能reshape

    def ref_masked_attention(self,
                             query,  # (1, num_heads, head_size)
                             key,  # (context_len, kv_heads, head_size)
                             value,
                             scale: float,
                             alibi_bias,
                             mask_data_type=torch.bfloat16
                             ):
        # Q * K.T
        query = query
        query = torch.permute(query, (1, 0, 2))
        if not self.is_int8_flag:
            key = torch.permute(key, (1, 2, 0))  # 0 1 2
        else:
            key = torch.permute(key, (1, 0, 2))
        sim_high = self.group_mm_torch(query.shape[0], key.shape[0], query, key, 1)  # (head_num, q_seqlen, k_seqlen)
        sim_out = sim_high.to(torch.float32)
        sim_high = sim_high.to(torch.float32) * scale
        if alibi_bias is not None:
            sim_high = sim_high + alibi_bias.to(torch.float32)
        # softmax
        p_high, lse = self.softmax_numpy(sim_high)
        p = torch.from_numpy(p_high).to(mask_data_type)
        p_high = torch.from_numpy(p_high)

        lse = torch.permute(torch.from_numpy(lse).to(mask_data_type), (1, 0, 2))  # (q_seqlen, head_num, 1)

        # P * V
        value = torch.permute(value, (1, 0, 2))
        out = self.group_mm_torch(query.shape[0], key.shape[0], p, value, 0)
        out_high = self.group_mm_torch(query.shape[0], key.shape[0], p_high, value, 0)
        out = torch.permute(out, (1, 0, 2))
        out_high = torch.permute(out_high, (1, 0, 2))
        sim_out = torch.permute(sim_out, (1, 0, 2))
        return out, out_high, sim_out, lse

    def ref_single_query_cached_kv_attention(self,
                                             sim,
                                             output,
                                             true_out,
                                             lse,        # (num_tokens, num_heads, 1)
                                             query,
                                             key_cache,  # (num_blocks, block_size, num_heads, head_size)
                                             value_cache,  # (num_blocks, block_size, num_heads, head_size)
                                             block_tables,
                                             context_lens,
                                             mask,
                                             mask_dim=4,
                                             mask_data_type=torch.bfloat16
                                             ) -> None:
        mask_index_coff = 1
        if self.compressHead:
            query = query.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size_qk)
            output = output.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size_vo)
            true_out = true_out.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads,
                                     self.head_size_vo)
            if mask_dim == 4:
                mask_shape = mask.shape
                mask = mask.view(mask_shape[0] * self.kv_heads, self.num_heads // self.kv_heads, 1,
                                 self.max_context_len)
            else:
                mask_index_coff = self.kv_heads
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size_qk = key_cache.shape[3]
        head_size_vo = value_cache.shape[3]
        block_size = value_cache.shape[1]

        num_input_tokens = query.shape[0]
        index = 0
        for i in range(len(context_lens)):
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            if context_len == 0:
                continue

            q = query[index].view(1, num_heads, head_size_qk)
            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)

                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            scale = np.float32(1.0 / (head_size_qk ** 0.5))
            if mask_dim == 4:
                out, out_high, sim_out, _ = self.ref_masked_attention(q, keys, values, scale,
                                                                         mask[i, :, :, :context_len], mask_data_type)
                out = out.reshape(num_heads, head_size_vo)
            elif mask_dim == 3:
                out, out_high, sim_out, _ = self.ref_masked_attention(q, keys, values, scale,
                                                                         mask[i // mask_index_coff, :, :context_len],
                                                                         mask_data_type)
                out = out.reshape(num_heads, head_size_vo)
            else:
                out, out_high, sim_out, lse_i = self.ref_masked_attention(q, keys, values, scale, mask,
                                                                          mask_data_type)
                out = out.reshape(num_heads, head_size_vo)
                lse_i = lse_i.reshape(num_heads, 1)
                lse[index] = lse_i.to(mask_data_type)
            out_high = out_high.reshape(num_heads, head_size_vo)
            sim_out = sim_out.reshape(1, num_heads * context_len)
            output[index] = out.to(mask_data_type)
            true_out[index] = out_high
            sim[index] = sim_out
            index = index + 1

    def calc_data(self, num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen,\
                  dtype, mask_dim = 0, mask_data_type = torch.bfloat16,\
                  dynamic_batch = False, dynamic_seqlen = None, is_int8_flag = False, has_bias = False,
                  compressHead = False, is_kv_combined = True, is_nz_in = False):
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.num_tokens = num_tokens
        self.compressHead = compressHead
        self.head_size_qk = head_size_qk
        self.head_size_vo = head_size_vo

        logging.debug(
            f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size_qk}, {head_size_vo}, {block_size}, {num_blocks}, {k_seqlen}, {dtype}')

        query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size_qk))).to(dtype)
        # (num_blocks, block_size, num_heads, head_size)
        kv_range = 1.0
        kv_type = dtype
        if is_int8_flag:
            kv_range = 4.0
            kv_type = torch.int8
        if not compressHead:
            key_cache = torch.from_numpy(
                np.random.uniform(-kv_range, kv_range, size=(num_blocks, block_size, kv_heads, head_size_qk))).to(
                kv_type)
            # (num_blocks, block_size, num_heads, head_size)
            if not is_kv_combined:
                value_cache = torch.from_numpy(
                    np.random.uniform(-kv_range, kv_range, size=(num_blocks, block_size, kv_heads, head_size_vo))).to(
                    kv_type)
            else:
                value_cache = key_cache[:, :, :, :head_size_vo]
        else:
            key_cache = torch.from_numpy(
                np.random.uniform(-kv_range, kv_range, size=(num_blocks * kv_heads, block_size, 1, head_size_qk))).to(
                kv_type)
            # (num_blocks, block_size, num_heads, head_size)
            if not is_kv_combined:
                value_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(
                num_blocks * kv_heads, block_size, 1, head_size_vo))).to(kv_type)
            else:
                value_cache = key_cache[:, :, :, :head_size_vo]
        self.data_type = dtype

        if dynamic_batch:
            context_lens = dynamic_seqlen
        else:
            context_lens = [k_seqlen] * num_tokens
        max_context_len = max(context_lens)
        self.max_context_len = max_context_len
        batch = len(context_lens)

        # alibi mask
        if mask_dim == 4:
            mask = np.zeros((batch, num_heads, 1, self.max_context_len), dtype=np.float32)
            alibi_slopes = self.get_alibi_slopes(num_heads)
            for i, context_len in enumerate(context_lens):
                if context_len == 0:
                    continue
                position_ids = np.arange(context_len).astype(np.int32)
                alibi_bias = (position_ids - context_len + 1).astype(np.float32)
                alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)  # (head_num, 1, context)
                mask[i, :, :, :context_len] = alibi_bias
            mask = torch.from_numpy(mask).to(mask_data_type)
        # normal mask
        elif mask_dim == 3:
            mask = np.zeros((batch, 1, max_context_len), dtype=np.float16)
            for i in range(batch):
                mask[i, :, :i] = -10000
            mask = torch.from_numpy(mask).to(mask_data_type)
        else:  # no mask
            mask = None

        if compressHead:
            context_lens = [val for val in context_lens for _ in range(kv_heads)]
        batch = len(context_lens)
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []  # （num_tokens, max_num_blocks_per_seq）
        for i in range(batch):
            block_table = [
                i * max_num_blocks_per_seq + _ for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)

        self.is_int8_flag = is_int8_flag
        if is_int8_flag:
            de_scale1_fp32 = np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
            de_scale1_int64 = self.process_deq_scale(de_scale1_fp32)

            de_scale2_fp32 = np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
            de_scale2_int64 = self.process_deq_scale(de_scale2_fp32)

            offset1 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)

            offset2 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)

            self.de_scale1_int64 = torch.tensor(list(de_scale1_int64), dtype=torch.int64)
            self.de_scale2_int64 = torch.tensor(list(de_scale2_int64), dtype=torch.int64)
            self.de_scale1_fp32 = torch.from_numpy(de_scale1_fp32)
            self.de_scale2_fp32 = torch.from_numpy(de_scale2_fp32)
            self.offset1 = torch.from_numpy(offset1)
            self.offset2 = torch.from_numpy(offset2)
            self.has_bias = has_bias

        shape_out = (num_tokens, num_heads, head_size_vo)
        ref_output = torch.zeros(shape_out, dtype=dtype)
        true_out = torch.zeros(shape_out, dtype=torch.float32)
        sim = torch.zeros((num_tokens, num_heads * k_seqlen), dtype=torch.float32)
        lse = torch.zeros((num_tokens, num_heads, 1), dtype=dtype)
        self.ref_single_query_cached_kv_attention(
            sim,
            ref_output,
            true_out,
            lse,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            mask,
            mask_dim,
            mask_data_type
        )

        self.q_split1, self.q_split2 = torch.split(query, [512, 64], dim=2)
        self.key_cache_split1, self.key_cache_split2 = torch.split(key_cache, [512, 64], dim=3)

        if (is_nz_in):
            key_cache_split1, key_cache_split2 = torch.split(key_cache, [512, 64], dim=3)
            key_cache_split1 = key_cache_split1.reshape(num_blocks, block_size, -1)
            key_cache_split2 = key_cache_split2.reshape(num_blocks, block_size, -1)
            key_cache_split1_nz = self.convert_nd_to_nz(key_cache_split1)
            key_cache_split2_nz = self.convert_nd_to_nz(key_cache_split2)
            self.key_cache_split1 = key_cache_split1_nz.to(mask_data_type).reshape(num_blocks, -1, block_size, 16)
            self.key_cache_split2 = key_cache_split2_nz.to(mask_data_type).reshape(num_blocks, -1, block_size, 16)

        self.block_tables = np.array(block_tables).astype(np.int32)
        self.contex_lens = np.array(context_lens).astype(np.int32)
        self.alib_mask = mask
        self.golden_out = ref_output
        self.true_out = true_out
        self.lse = lse

    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out)
        return [golden_out, self.lse]

    def golden_compare(self, out_tensors, golden_tensors):
        go_double = compare_cv(self.true_out, golden_tensors[0], out_tensors[0])
        go_old = self.compare_output_data(out_tensors[0], golden_tensors[0], [0.001, 0.001, 0.005, 0.005])

        lse_double = True
        lse_old = True
        if self.is_ring:
            lse_double = compare_cv(golden_tensors[1], golden_tensors[1], out_tensors[1])
            lse_old = self.compare_output_data(out_tensors[1], golden_tensors[1], [0.001, 0.001, 0.005, 0.005])

        return (go_double or go_old) and (lse_double or lse_old)

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_128_fp16(self):
        self.set_support_910b_only()
        num_tokens = 32
        q_seqlen_list = [1] * num_tokens
        k_seqlen_list = [256] * num_tokens
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 256
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        self.is_ring = 0
        is_nz_in = False

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen,
                       dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in = is_nz_in)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads,
            "tor": tor, "qSeqLen": q_seqlen_list, "kvSeqLen": k_seqlen_list, "maskType": 0, "isRing": self.is_ring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
                      f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=dtype)

        for i in range(1):
            self.execute(
                [
                    self.q_split1,
                    self.q_split2,
                    self.key_cache_split1,
                    self.key_cache_split2,
                    torch.tensor(self.block_tables).int(),
                    torch.tensor([], dtype=dtype),
                    torch.tensor([1], dtype=torch.float),
                    torch.tensor([1], dtype=torch.float)
                ],
                [
                    attention_out, torch.tensor([])
                ]
            )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_tp1_fp16(self):
        self.set_support_910b_only()
        num_tokens = 27
        q_seqlen_list = [1] * num_tokens
        k_seqlen_list = [1150] * num_tokens
        num_heads = 128
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 243
        k_seqlen = 1150
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        self.is_ring = 0
        is_nz_in = False

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen,
                       dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in = is_nz_in)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads,
            "tor": tor, "qSeqLen": q_seqlen_list, "kvSeqLen": k_seqlen_list, "maskType": 0, "isRing": self.is_ring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
                      f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=dtype)

        for i in range(1):
            self.execute(
                [
                    self.q_split1,
                    self.q_split2,
                    self.key_cache_split1,
                    self.key_cache_split2,
                    torch.tensor(self.block_tables).int(),
                    torch.tensor([], dtype=dtype),
                    torch.tensor([1], dtype=torch.float),
                    torch.tensor([1], dtype=torch.float)
                ],
                [
                    attention_out, torch.tensor([])
                ]
            )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_128_ring_fp16(self):
        self.set_support_910b_only()
        num_tokens = 32
        q_seqlen_list = [1] * num_tokens
        k_seqlen_list = [256] * num_tokens
        num_heads = 128
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 256
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        self.is_ring = 1
        is_nz_in = False

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen,
                       dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in = is_nz_in)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads,
        "tor": tor, "qSeqLen": q_seqlen_list, "kvSeqLen": k_seqlen_list, "maskType": 0, "isRing": self.is_ring}

        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
                      f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=dtype)

        shape_out_2 = ((num_tokens, num_heads, 1))
        lse = torch.zeros(shape_out_2, dtype=dtype)

        for i in range(1):
            self.execute(
                [
                    self.q_split1,
                    self.q_split2,
                    self.key_cache_split1,
                    self.key_cache_split2,
                    torch.tensor(self.block_tables).int(),
                    torch.tensor([], dtype=dtype),
                    torch.tensor([1], dtype=torch.float),
                    torch.tensor([1], dtype=torch.float)
                ],
                [
                    attention_out, lse
                ]
            )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_128_bf16(self):
        self.set_support_910b_only()
        num_tokens = 32
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 256
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.bfloat16
        is_kv_combined = True
        self.is_ring = 0
        is_nz_in = False

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen,
                       dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in = is_nz_in)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead": kv_heads, "headSize": num_heads, "tor": tor,
                    "kvSeqLen": self.contex_lens.tolist(), "isRing": self.is_ring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
                      f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=dtype)

        for i in range(1):
            self.execute(
                [
                    self.q_split1,
                    self.q_split2,
                    self.key_cache_split1,
                    self.key_cache_split2,
                    torch.tensor(self.block_tables).int(),
                    torch.tensor([], dtype=dtype),
                    torch.tensor([1], dtype=torch.float),
                    torch.tensor([1], dtype=torch.float)
                ],
                [
                    attention_out, torch.tensor([])
                ]
            )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_128_ring_bf16(self):
        self.set_support_910b_only()
        num_tokens = 32
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 256
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.bfloat16
        is_kv_combined = True
        self.is_ring = 1
        is_nz_in = False

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen,
                       dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in = is_nz_in)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead": kv_heads, "headSize": num_heads, "tor": tor,
                    "kvSeqLen": self.contex_lens.tolist(), "isRing": self.is_ring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
                      f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=dtype)

        shape_out_2 = ((num_tokens, num_heads, 1))
        lse = torch.zeros(shape_out_2, dtype=dtype)

        for i in range(1):
            self.execute(
                [
                    self.q_split1,
                    self.q_split2,
                    self.key_cache_split1,
                    self.key_cache_split2,
                    torch.tensor(self.block_tables).int(),
                    torch.tensor([], dtype=dtype),
                    torch.tensor([1], dtype=torch.float),
                    torch.tensor([1], dtype=torch.float)
                ],
                [
                    attention_out, lse
                ]
            )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_128_nz(self):
        self.set_support_910b_only()
        num_tokens = 32
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 256
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        self.is_ring = 0
        is_nz_in = True

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen,
                       dtype, mask_dim, dtype,
                       is_kv_combined=is_kv_combined, is_nz_in=True)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead": kv_heads, "headSize": num_heads, "tor": tor,
                    "kvSeqLen": self.contex_lens.tolist(), "isRing": self.is_ring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
                      f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        logging.info(f"Q1 shape: {self.q_split1.shape}")
        logging.info(f"Q2 shape: {self.q_split2.shape}")
        logging.info(f"K1 shape: {self.key_cache_split1.shape}")
        logging.info(f"K2 shape: {self.key_cache_split2.shape}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=torch.float16)
        for i in range (1):
            self.execute(
            [
                self.q_split1,
                self.q_split2,
                torch.tensor(self.key_cache_split1),
                torch.tensor(self.key_cache_split2),
                torch.tensor(self.block_tables).int(),
                torch.tensor([], dtype=dtype),
                torch.tensor([1], dtype=torch.float),
                torch.tensor([1], dtype=torch.float)
            ],
            [
                attention_out, torch.tensor([])
            ]
        )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_128_nz_non16(self):
        self.set_support_910b_only()
        num_tokens = 32
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 129
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        self.is_ring = 0
        is_nz_in = True

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, dtype,
                        is_kv_combined = is_kv_combined, is_nz_in = True)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead": kv_heads, "headSize": num_heads, "tor": tor,
            "kvSeqLen": self.contex_lens.tolist(), "isRing": self.is_ring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
              f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        logging.info(f"Q1 shape: {self.q_split1.shape}")
        logging.info(f"Q2 shape: {self.q_split2.shape}")
        logging.info(f"K1 shape: {self.key_cache_split1.shape}")
        logging.info(f"K2 shape: {self.key_cache_split2.shape}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=torch.float16)
        for i in range (1):
            self.execute(
            [
                self.q_split1,
                self.q_split2,
                torch.tensor(self.key_cache_split1),
                torch.tensor(self.key_cache_split2),
                torch.tensor(self.block_tables).int(),
                torch.tensor([], dtype=dtype),
                torch.tensor([1], dtype=torch.float),
                torch.tensor([1], dtype=torch.float)
            ],
            [
                attention_out, torch.tensor([])
            ]
        )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_bf16(self):
        self.set_support_910b_only()
        num_tokens = 1
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 129
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.bfloat16
        is_kv_combined = True
        self.is_ring = 0
        is_nz_in = False

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, dtype,
                        is_kv_combined = is_kv_combined)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads, "tor": tor, "kvSeqLen": self.contex_lens.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
              f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        logging.info(f"Q1 shape: {self.q_split1.shape}")
        logging.info(f"Q2 shape: {self.q_split2.shape}")
        logging.info(f"K1 shape: {self.key_cache_split1.shape}")
        logging.info(f"K2 shape: {self.key_cache_split2.shape}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=dtype)
        for i in range (1):
            self.execute(
            [
                self.q_split1,
                self.q_split2,
                torch.tensor(self.key_cache_split1),
                torch.tensor(self.key_cache_split2),
                torch.tensor(self.block_tables).int(),
                torch.tensor([], dtype=dtype),
                torch.tensor([1], dtype=torch.float),
                torch.tensor([1], dtype=torch.float)
            ],
            [
                attention_out, torch.tensor([])
            ]
        )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_128_nz_bf16(self):
        self.set_support_910b_only()
        num_tokens = 1
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = 257
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.bfloat16
        is_kv_combined = True
        self.is_ring = 0
        is_nz_in = True

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, dtype,
                        is_kv_combined = is_kv_combined, is_nz_in = True)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads, "tor": tor, "kvSeqLen": self.contex_lens.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
              f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        logging.info(f"Q1 shape: {self.q_split1.shape}")
        logging.info(f"Q2 shape: {self.q_split2.shape}")
        logging.info(f"K1 shape: {self.key_cache_split1.shape}")
        logging.info(f"K2 shape: {self.key_cache_split2.shape}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=dtype)
        for i in range (1):
            self.execute(
            [
                self.q_split1,
                self.q_split2,
                torch.tensor(self.key_cache_split1),
                torch.tensor(self.key_cache_split2),
                torch.tensor(self.block_tables).int(),
                torch.tensor([], dtype=dtype),
                torch.tensor([1], dtype=torch.float),
                torch.tensor([1], dtype=torch.float)
            ],
            [
                attention_out, torch.tensor([])
            ]
        )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_32head_opt(self):
        self.set_support_910b_only()
        num_tokens = 1
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 500
        k_seqlen = 1024
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        self.is_ring = 0
        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, torch.float16,
                        is_kv_combined = is_kv_combined)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads, "tor": tor, "kvSeqLen": self.contex_lens.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
              f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        logging.info(f"Q1 shape: {self.q_split1.shape}")
        logging.info(f"Q2 shape: {self.q_split2.shape}")
        logging.info(f"K1 shape: {self.key_cache_split1.shape}")
        logging.info(f"K2 shape: {self.key_cache_split2.shape}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=torch.float16)
        for i in range (1):
            self.execute(
            [
                self.q_split1,
                self.q_split2,
                torch.tensor(self.key_cache_split1),
                torch.tensor(self.key_cache_split2),
                torch.tensor(self.block_tables).int(),
                torch.tensor([], dtype=torch.float16),
                torch.tensor([1], dtype=torch.float),
                torch.tensor([1], dtype=torch.float)
            ],
            [
                attention_out, torch.tensor([])
            ]
        )

    @op_test.only_910b
    def test_paged_mla_combine_cache_norm_32head_opt_nz(self):
        self.set_support_910b_only()
        num_tokens = 1
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 500
        k_seqlen = 1025
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        is_nz_in = True
        self.is_ring = 0


        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, torch.float16,
                        is_kv_combined = is_kv_combined, is_nz_in = True)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads, "tor": tor, "kvSeqLen": self.contex_lens.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
              f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        logging.info(f"Q1 shape: {self.q_split1.shape}")
        logging.info(f"Q2 shape: {self.q_split2.shape}")
        logging.info(f"K1 shape: {self.key_cache_split1.shape}")
        logging.info(f"K2 shape: {self.key_cache_split2.shape}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=torch.float16)
        for i in range (1):
            self.execute(
            [
                self.q_split1,
                self.q_split2,
                torch.tensor(self.key_cache_split1),
                torch.tensor(self.key_cache_split2),
                torch.tensor(self.block_tables).int(),
                torch.tensor([], dtype=torch.float16),
                torch.tensor([1], dtype=torch.float),
                torch.tensor([1], dtype=torch.float)
            ],
            [
                attention_out, torch.tensor([])
            ]
        )

    @op_test.only_910b
    def test_paged_mla_combine_cache_deepseek_test1(self):
        self.set_support_910b_only()
        num_tokens = 32
        num_heads = 128
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 801
        k_seqlen = 3073
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        is_nz_in = False
        self.is_ring = 0

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, torch.float16,
                        is_kv_combined = is_kv_combined, is_nz_in = is_nz_in)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads, "tor": tor, "kvSeqLen": self.contex_lens.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
              f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        logging.info(f"Q1 shape: {self.q_split1.shape}")
        logging.info(f"Q2 shape: {self.q_split2.shape}")
        logging.info(f"K1 shape: {self.key_cache_split1.shape}")
        logging.info(f"K2 shape: {self.key_cache_split2.shape}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=torch.float16)
        for i in range (1):
            self.execute(
            [
                self.q_split1,
                self.q_split2,
                torch.tensor(self.key_cache_split1),
                torch.tensor(self.key_cache_split2),
                torch.tensor(self.block_tables).int(),
                torch.tensor([], dtype=torch.float16),
                torch.tensor([1], dtype=torch.float),
                torch.tensor([1], dtype=torch.float)
            ],
            [
                attention_out, torch.tensor([])
            ]
        )
    
    @op_test.only_910b
    def test_paged_mla_combine_cache_deepseek_test2(self):
        self.set_support_910b_only()
        num_tokens = 32
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 801
        k_seqlen = 3073
        tor = 1.0 / (head_size_qk ** 0.5)
        mask_dim = 0
        dtype = torch.float16
        is_kv_combined = True
        is_nz_in = False
        self.is_ring = 0

        self.calc_data(num_tokens, num_heads, kv_heads, head_size_qk, head_size_vo, block_size, num_blocks, k_seqlen, dtype, mask_dim, torch.float16,
                        is_kv_combined = is_kv_combined, is_nz_in = is_nz_in)

        OP_NAME = "MLAOperation"
        OP_PARAM = {"type": 0, "kvHead":kv_heads, "headSize":num_heads, "tor": tor, "kvSeqLen": self.contex_lens.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
              f", blockSize: {block_size}, headSizeQK: {head_size_qk}, headSizeVO: {head_size_vo}, numBlocks: {num_blocks}")
        logging.info(f"Q1 shape: {self.q_split1.shape}")
        logging.info(f"Q2 shape: {self.q_split2.shape}")
        logging.info(f"K1 shape: {self.key_cache_split1.shape}")
        logging.info(f"K2 shape: {self.key_cache_split2.shape}")
        shape_out = ((num_tokens, num_heads, head_size_vo))
        attention_out = torch.zeros(shape_out, dtype=torch.float16)
        for i in range (1):
            self.execute(
            [
                self.q_split1,
                self.q_split2,
                torch.tensor(self.key_cache_split1),
                torch.tensor(self.key_cache_split2),
                torch.tensor(self.block_tables).int(),
                torch.tensor([], dtype=torch.float16),
                torch.tensor([1], dtype=torch.float),
                torch.tensor([1], dtype=torch.float)
            ],
            [
                attention_out, torch.tensor([])
            ]
        )

if __name__ == '__main__':
    unittest.main()
