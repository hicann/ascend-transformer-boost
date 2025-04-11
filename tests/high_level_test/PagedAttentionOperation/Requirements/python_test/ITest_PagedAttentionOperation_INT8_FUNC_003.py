# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendOpCommonLib is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# 

import logging
import sys,os
import unittest
import math
import numpy as np
sys.path.append('../../')
sys.path.append('../../..')
import torch
import random
import sys
import numpy as np
import math
np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024
# from paged_attention.precision_calcu import *
# from precision_calcu import *

import json
import torch_npu
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import operation_test  # NOQA: E402

class TestPagedAttentionDataGenerator():

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
        logging.info("1/1000 Accuracy is %f",  1 - float(error_count) / len)
        logging.info("5/1000 Accuracy is %f",  1 - float(strict_error_count) / len)
        if self.data_type == torch.bfloat16 or self.is_int8_flag:
            logging.info("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[2])
        else:
            logging.info("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[0])
        calc_times = self.head_size * self.max_context_len + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else :
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.info("accuracy is correct in new standard: %r", res)
            return res
        else:
            if calc_times < 2048:
                error = 2**(-8)
            else :
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.info("accuracy is correct in new standard: %r", res)
            return res
 
    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** ( -4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        # slopes = torch.ones(n_heads)
        return slopes
 
    def group_mm_torch(self, heads, group_num, A, B, razor_mod, is_k):
        group_head = heads // group_num
        score_high = None
        for i in range(group_num):
            if self.is_int8_flag:
                int8_B = B[i: (i+1), :, :, ]
                head_dim = int8_B.shape[2]
                int32_B = torch.matmul(torch.eye(int8_B.shape[1]).to(torch.float32), int8_B.to(torch.float32)).to(torch.int32)
                if is_k:
                    if self.has_bias:
                        int32_B = int32_B + self.offset1[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                    fp32_B = int32_B.to(torch.float32) * self.de_scale1_fp32[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                    fp32_B = torch.permute(fp32_B, (0, 2, 1))
                else:
                    if self.has_bias:
                        int32_B = int32_B + self.offset2[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                    fp32_B = int32_B.to(torch.float32) * self.de_scale2_fp32[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                            fp32_B)
            elif self.is_quant_flag:
                    group_score_int32 = torch.matmul(A[i*group_head: (i+1)*group_head, :, :].to(torch.int32),
                        B[i: (i+1), :, :].to(torch.int32)).to(torch.int32)
                    if is_k:
                        group_score_high = group_score_int32.to(torch.float32) * self.de_scale1_fp32[i*group_head: (i+1)*group_head].reshape(group_head, 1, 1).to(torch.float32)
                    else:
                        group_score_high = group_score_int32.to(torch.float32) * self.de_scalev[i*group_head: (i+1)*group_head].reshape(group_head, 1, 1).to(torch.float32)
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
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res
 
    def softmax_quant_numpy(self, sim, gm, is_first):
        lm = np.max(sim, axis=-1, keepdims=True)
        hm = np.maximum(gm, lm)
        if is_first:
            gm = hm
        dm = gm - hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        row_maxp = np.max(sim_sub, axis=-1, keepdims=True)
        if not self.is_quant_offiline:
            scale = row_maxp.astype("float32") / 127.0
            sim_int8 = sim_sub / scale
            soft_res = sim_int8.astype("float16")
            soft_res = np.rint(soft_res).astype("int8")
            de_scalev = self.de_scale2_fp32 * row_maxp[:,0,0] / 127
        else:
            soft_res = sim_sub * self.scale.reshape(self.scale.shape[0], 1, 1).numpy()
            soft_res = soft_res.astype("float16")
            soft_res = np.rint(soft_res).astype("int8")
            de_scalev = self.de_scale2_fp32
        return soft_res, row_sum, de_scalev, hm, dm
 
    def softmax_quant_numpy_online(self, sim, heads, kv_head, value, razor_mod):
        group_head = heads // kv_head
        score_high = None
        # (kv_heads, context_len, head_size)
        kv_seqlen = value.shape[1]
        cur_kv_seqlen = self.kv_split_per_core
        gm = np.full([heads , 1, 1],  np.finfo(np.float32).min)
        hm = np.full([heads , 1, 1],  np.finfo(np.float32).min)
        for cur_nIndx in range(self.kvsplit):
            kv_seqlen_align =  (kv_seqlen + self.block_size - 1) // self.block_size  * self.block_size
            start_kv = cur_nIndx * self.kv_split_per_core
            cur_kv_seqlen = self.kv_split_per_core
            kv_loop = (kv_seqlen_align + self.kv_split_per_core - 1) // self.kv_split_per_core
            if cur_nIndx >= kv_loop:
                continue
            if cur_nIndx == (kv_loop - 1):
                cur_kv_seqlen = kv_seqlen - cur_nIndx * self.kv_split_per_core
            n_loop = (cur_kv_seqlen + self.block_size_calc - 1) // self.block_size_calc
            qk_n = self.block_size_calc
            end_kv = start_kv
            for n_idx in range(n_loop):
                is_first = (cur_nIndx == 0) and (n_idx == 0)
                if n_idx == n_loop - 1:
                    qk_n = cur_kv_seqlen - n_idx * self.block_size_calc
                end_kv = end_kv + qk_n
                sim_block = sim[:, :, start_kv : end_kv]
                p_block, row_sum, de_scalev, hm, dm = self.softmax_quant_numpy(sim_block, gm, is_first)
                self.de_scalev = de_scalev
                value_block = value[:, start_kv : end_kv, :]
                lo = self.group_mm_torch(heads, kv_head, torch.from_numpy(p_block),  value_block, razor_mod, 0)
                lo = lo.cpu().numpy()
                gm = hm
                dm = np.exp(dm)
                gl = row_sum * dm
                gl = gl + row_sum
                if cur_nIndx == 0 and n_idx == 0:
                    go = lo
                else:
                    go = go * dm
                    go = go + lo
            go = go / gl
            return torch.from_numpy(go)
 
    def ref_masked_attention(self,
            query,  # (1, num_heads, head_size)
            key,  # (context_len, kv_heads, head_size)
            value,
            scale: float,
            alibi_bias,
            razor_rope,
            razor_offset_list,
            razor_mod,
            mask_data_type = torch.bfloat16
    ):
        # Q * K.T
        query = query
        query = torch.permute(query, (1, 0, 2))
        if not self.is_int8_flag:
            key = torch.permute(key, (1, 2, 0))  # 0 1 2
        else:
            key = torch.permute(key, (1, 0, 2))
        sim_high = self.group_mm_torch(query.shape[0], key.shape[0], query, key, razor_mod, 1)  # (head_num, q_seqlen, k_seqlen)
 
        if razor_rope:
            razor_offset_list = razor_offset_list.view(1, 1, razor_offset_list.shape[0])
            sim_high = sim_high.to(torch.float32) + razor_offset_list
 
        sim_high = sim_high.to(torch.float32) * scale
        if alibi_bias is not None:
            sim_high = sim_high + alibi_bias.to(torch.float32)
        # softmax
 
 
 
 
        if self.is_quant_flag:
            p_high, row_sum, de_scalev, _, _ = self.softmax_quant_numpy(sim_high.numpy(), np.full([query.shape[0] , 1, 1],  np.finfo(np.float32).min), 1)
            self.de_scalev = de_scalev
            value = torch.permute(value, (1, 0, 2))
            out_high = self.group_mm_torch(query.shape[0], key.shape[0], torch.from_numpy(p_high), value, razor_mod, 0)
            out_high = out_high / row_sum
            out_high = torch.permute(out_high, (1, 0, 2))
            s_qk = sim_high.numpy()
            out = self.softmax_quant_numpy_online(s_qk, query.shape[0], key.shape[0], value, razor_mod)
            out = out_high
        else:
            p_high = self.softmax_numpy(sim_high)
            p = torch.from_numpy(p_high).to(mask_data_type)
            p_high = torch.from_numpy(p_high)
            # P * V
            value = torch.permute(value, (1, 0, 2))
            out = self.group_mm_torch(query.shape[0], key.shape[0], p, value, razor_mod, 0)
            out_high = self.group_mm_torch(query.shape[0], key.shape[0], p_high, value, razor_mod, 0)
            out = torch.permute(out, (1, 0, 2))
            out_high = torch.permute(out_high, (1, 0, 2))
        return out, out_high
 
    def ref_single_query_cached_kv_attention(self,
            output,
            true_out,
            query,
            key_cache,  # (num_blocks, block_size, num_heads, head_size)
            value_cache,  # (num_blocks, block_size, num_heads, head_size)
            block_tables,
            context_lens,
            mask,
            razor_offset,
            razor_rope,
            mask_dim = 4,
            mask_data_type = torch.bfloat16    
    ) -> None:
        mask_index_coff = 1
        if self.compressHead:
            query = query.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size)
            output = output.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size)
            true_out = true_out.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size)
            if mask_dim == 4:
                mask_shape = mask.shape
                mask = mask.view(mask_shape[0] * self.kv_heads, self.num_heads // self.kv_heads, 1, self.max_context_len)
            else:
                mask_index_coff = self.kv_heads
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size = value_cache.shape[3]
        block_size = value_cache.shape[1]
 
        num_input_tokens = query.shape[0]
        index = 0
        razor_mod = 0
        for i in range(len(context_lens)):
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            if context_len == 0:
                continue
 
            q = query[index].view(1, num_heads, head_size)
            keys = []
            values = []
            razor_offset_list = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
 
                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size)
                keys.append(k)
 
                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size)
                values.append(v)
 
                if razor_rope:
                    offset = razor_offset[block_number, block_offset]
                    razor_offset_list.append(offset)
      
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            if razor_rope:
                razor_mod = i % self.kv_heads
                razor_offset_list = torch.stack(razor_offset_list, axis=0)
 
            if self.compressHead:
                razor_mod = i % self.kv_heads
            
            scale = np.float32(1.0 / (head_size ** 0.5))
            if mask_dim == 4:
                out, out_high = self.ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len], razor_rope, razor_offset_list, razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size)
            elif mask_dim == 3:
                out,out_high = self.ref_masked_attention(q, keys, values, scale, mask[i // mask_index_coff, :, :context_len], razor_rope, razor_offset_list, razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size)
            else:
                out,out_high = self.ref_masked_attention(q, keys, values, scale, mask, razor_rope, razor_offset_list, razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size)
            out_high = out_high.reshape(num_heads, head_size)
            output[index] = out.to(mask_data_type)
            true_out[index] = out_high
            index = index + 1
 
    def get_blockszie_calc(self, max_context_len, block_size, embeddingSize, embeddingSizeV):
        embedQKSplit = 256 if embeddingSize > 256 else embeddingSize
        embedVOSplit = 256 if embeddingSizeV > 256 else embeddingSizeV
        BLOCK_LIMIT = 128 * 128
        KV_SEQLEN_SLICE = 128
        KV_SEQLEN_SLICE_256 = 256
        KV_SEQLEN_SLICE_512 = 512
        BLOCK_LIMIT_NO_PINGPONG = 128 * 256;
        block_size_calc = block_size
        headdimMax =  np.maximum(embedQKSplit, embedVOSplit)
        if block_size <= KV_SEQLEN_SLICE / 2 and \
            block_size * 2 * embedQKSplit <= BLOCK_LIMIT and \
            block_size * 2 * embedVOSplit <= BLOCK_LIMIT:
            block_size_calc =  block_size * 2
        if not self.is_int8_flag and \
            max_context_len >= KV_SEQLEN_SLICE_256 and \
            self.kv_split_per_core >= KV_SEQLEN_SLICE_256 and \
            KV_SEQLEN_SLICE_256 * embedQKSplit  <= BLOCK_LIMIT_NO_PINGPONG and \
            KV_SEQLEN_SLICE_256 * embedVOSplit <= BLOCK_LIMIT_NO_PINGPONG and \
            (block_size == KV_SEQLEN_SLICE_256 // 4 or block_size ==  KV_SEQLEN_SLICE_256 // 2):
            block_size_calc = 256
       
        if self.is_quant_flag and \
            max_context_len >= KV_SEQLEN_SLICE_512 and \
            self.kv_split_per_core >= KV_SEQLEN_SLICE_512 and \
            KV_SEQLEN_SLICE_512 * embedQKSplit  <= BLOCK_LIMIT_NO_PINGPONG * 2 and \
            KV_SEQLEN_SLICE_512 * embedVOSplit <= BLOCK_LIMIT_NO_PINGPONG * 2 and \
            (block_size == KV_SEQLEN_SLICE_256 // 4 or block_size ==  KV_SEQLEN_SLICE_256 // 2) and \
            KV_SEQLEN_SLICE_512 * headdimMax <= BLOCK_LIMIT_NO_PINGPONG and self.head_num_move < 4:
            block_size_calc = KV_SEQLEN_SLICE_512
        return block_size_calc
 
    def getkvsplit(self, num_tokens, num_heads, max_context_len, block_size, blocknum, isLongSeq):
        if isLongSeq:
            kvSeqklenMaxAlign = (max_context_len + block_size - 1) // block_size * block_size
            kvSeqBlockNum = int(kvSeqklenMaxAlign / block_size)
            kvBlockPreCore = int((kvSeqBlockNum + blocknum - 1)) // blocknum
            kvSplitPerCore = int(kvBlockPreCore * block_size)
            kvSplitCoreNum = int(kvSeqklenMaxAlign + kvSplitPerCore - 1) // kvSplitPerCore
            headSplit = int((num_heads + kvSplitCoreNum - 1) // kvSplitCoreNum)
        else:
            coreNumPerBatch  = int((blocknum + num_tokens - 1) // num_tokens)
            kvSeqklenMaxAlign = (max_context_len + block_size - 1) // block_size * block_size
            kvSeqBlockNum = int(kvSeqklenMaxAlign / block_size)
            kvBlockPreCore = int((kvSeqBlockNum + coreNumPerBatch - 1)) // coreNumPerBatch
            kvSplitPerCore = int(kvBlockPreCore * block_size)
            kvSplitCoreNum = int(kvSeqklenMaxAlign + kvSplitPerCore - 1) // kvSplitPerCore
            headSplit = int((num_heads + kvSplitCoreNum - 1) // kvSplitCoreNum)
        return kvSplitCoreNum, kvSplitPerCore
    
    def get_head_num_move(self, num_heads, kvhead, embeddingSize, embeddingSizeV):
        if embeddingSize % 32 == 0 and embeddingSizeV % 32 == 0 and embeddingSize <= 128 and embeddingSizeV <= 128 and num_heads == kvhead:
            head_num_move = 4
        else:
            head_num_move = 1
        return head_num_move
 
    def calc_data(self, num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen,\
                  dtype, mask_dim = 4, mask_data_type = torch.bfloat16,\
                  dynamic_batch = False, dynamic_seqlen = None, is_int8_flag = False, has_bias = False,
                  compressHead = False, razor_rope = False, blocknum = 20, is_quant_flag = 0, is_quant_offiline = 0):
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.num_tokens = num_tokens
        self.compressHead = compressHead
        self.head_size = head_size
 
        logging.debug(f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size}, {block_size}, {num_blocks}, {k_seqlen}, {dtype}')
 
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        kv_type = dtype
        self.is_quant_flag = is_quant_flag
        self.is_quant_offiline = is_quant_offiline
        if self.is_quant_flag:
            q_min_range = 0
            q_max_range =  5
            kv_min_range = 0
            kv_max_range =  5
            dtype = torch.int8
            kv_type = torch.int8
        if is_int8_flag:
            kv_min_range = -4
            kv_max_range =  4
            kv_type = torch.int8
        query = torch.from_numpy(np.random.uniform(q_min_range, q_max_range, size=(num_tokens, num_heads, head_size))).to(dtype)
        # (num_blocks, block_size, num_heads, head_size)
        if not compressHead:
            key_cache = torch.from_numpy(np.random.uniform(kv_min_range, kv_max_range, size=(num_blocks, block_size, kv_heads, head_size))).to(kv_type)
            # # (num_blocks, block_size, num_heads, head_size)
            value_cache = torch.from_numpy(np.random.uniform(kv_min_range, kv_max_range, size=(num_blocks, block_size, kv_heads, head_size))).to(kv_type)
            # (num_blocks, block_size, num_heads, head_size)
 
        else:
            key_cache = torch.from_numpy(np.random.uniform(kv_min_range, kv_max_range, size=(num_blocks * kv_heads, block_size, 1, head_size))).to(kv_type)
            # # (num_blocks, block_size, num_heads, head_size)
            value_cache = torch.from_numpy(np.random.uniform(kv_min_range, kv_max_range, size=(num_blocks * kv_heads, block_size, 1, head_size))).to(kv_type)
            # (num_blocks, block_size, num_heads, head_size)
 
        self.data_type = dtype
 
        razor_offset = torch.tensor([], dtype=torch.float32)
        if razor_rope:
            razor_offset = torch.zeros(num_blocks * kv_heads, block_size)
            mask = np.random.choice([False, True], size=num_blocks * kv_heads, p=[0.2, 0.8])
 
            random_indices = np.random.randint(0, block_size, size=np.sum(mask))
            random_values = np.random.uniform(0, 20, size=np.sum(mask))
 
            active_rows = np.where(mask)[0]
            razor_offset[active_rows, random_indices] = torch.from_numpy(random_values).to(torch.float32)
        
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
                alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
                mask[i, :, :, :context_len] = alibi_bias
            mask = torch.from_numpy(mask).to(mask_data_type)
        # normal mask
        elif mask_dim == 3:
            mask = np.zeros((batch, 1, max_context_len), dtype=np.float16)
            for i in range(batch):
                mask[i, :, :i] = -10000
            mask = torch.from_numpy(mask).to(mask_data_type)
        else: # no mask
            mask = None
 
        if compressHead:
            context_lens = [val for val in context_lens for _ in range(kv_heads)]
        batch = len(context_lens)
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []   # （num_tokens, max_num_blocks_per_seq）
        for _ in range(batch):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
 
        self.is_int8_flag = is_int8_flag
 
        if is_int8_flag:
            de_scale1_fp32 = np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
            de_scale1_int64 = self.process_deq_scale(de_scale1_fp32)
 
            de_scale2_fp32 =  np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
            de_scale2_int64 = self.process_deq_scale(de_scale2_fp32)
 
            offset1 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)
 
            offset2 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)
 
            self.de_scale1_int64 = torch.tensor(list(de_scale1_int64), dtype=torch.int64)
            self.de_scale2_int64 =  torch.tensor(list(de_scale2_int64), dtype=torch.int64)
            self.de_scale1_fp32 = torch.from_numpy(de_scale1_fp32)
            self.de_scale2_fp32 = torch.from_numpy(de_scale2_fp32)
            self.offset1 = torch.from_numpy(offset1)
            self.offset2 = torch.from_numpy(offset2)
            self.has_bias = has_bias
 
        if self.is_quant_flag:
            self.de_scale1_fp32 = torch.from_numpy(np.random.uniform(0, 1/127, size=(num_heads)).astype(np.float32)).to(torch.float32)
            self.de_scale2_fp32 =  torch.from_numpy(np.random.uniform(0, 1/127, size=(num_heads)).astype(np.float32)).to(torch.float32)
            self.scale = torch.from_numpy(np.random.uniform(0, 5/127, size=(num_heads)).astype(np.float32)).to(torch.float32)
            isLongSeq = max_context_len > blocknum * 128 * 2 and num_tokens < blocknum * 0.8
            if num_tokens * num_heads < 0.8 * blocknum or isLongSeq:
                self.kvsplit, self.kv_split_per_core = self.getkvsplit(num_tokens, num_heads, max_context_len, block_size, blocknum, isLongSeq)
            else:
                self.kvsplit = 1
                self.kv_split_per_core = max_context_len
            self.head_num_move = self.get_head_num_move(num_heads, kv_heads, head_size, head_size)
            self.block_size_calc = self.get_blockszie_calc(max_context_len, block_size, head_size, head_size)
            self.block_size = block_size
 
        ref_output = torch.zeros_like(query).to(torch.float32)
        true_out = torch.zeros_like(query, dtype=torch.float32)
 
        self.ref_single_query_cached_kv_attention(
            ref_output,
            true_out,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            mask,
            razor_offset,
            razor_rope,
            mask_dim,
            mask_data_type 
        )
 
        self.q = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = np.array(block_tables).astype(np.int32)
        self.contex_lens = np.array(context_lens).astype(np.int32)
        self.alib_mask = mask
        self.golden_out = ref_output
        self.true_out = true_out
        self.razor_offset = razor_offset
 
    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out)
        return [golden_out]
 
    def golden_compare(self, out_tensors, golden_tensors):
        logging.debug(f"out_tensors: {out_tensors}")
        logging.debug(f"golden_tensors:{golden_tensors}")
        result_double = compare_cv(golden_tensors[0], self.true_out, out_tensors[0])
        result_old = self.compare_output_data(out_tensors[0], golden_tensors[0], [0.001, 0.001, 0.005, 0.005])
        return (result_double or result_old)

    def test_pa_quant_case_normal_mask(self):
        # self.set_support_910b_only()
        num_tokens = 9
        num_heads = 32
        kv_heads = 2
        block_size = 128
        head_size = 128
        num_blocks = 64 
        dynamic_batch = True
        batch_tatus = [1] * num_tokens
        k_seqlen = [3000, 300, 14000, 33, 65, 1, 16, 14000, 300]
        tor = 1.0 / (head_size ** 0.5)
        dtype = torch.bfloat16
        outDtype = torch.bfloat16
        mask_dim = 3
        is_quant_flag = 1
        self.calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype, mask_dim, outDtype, dynamic_batch, k_seqlen, is_quant_flag = is_quant_flag)
        attention_out = torch.zeros_like(self.q).to(outDtype)
        attention_out[:] = 0.1
        
    def test_pa_quantoffline_case_normal_mask(self):
        # self.set_support_910b_only()
        num_tokens = 9
        num_heads = 32
        kv_heads = 2
        block_size = 128
        head_size = 128
        num_blocks = 64 
        dynamic_batch = True
        batch_tatus = [1] * 9
        k_seqlen = [3000, 300, 14000, 33, 65, 1, 16, 14000, 300]
        tor = 1.0 / (head_size ** 0.5)
        dtype = torch.float16
        outDtype = torch.float16
        mask_dim = 3
        is_quant_flag = 1
        self.calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype, mask_dim, outDtype, dynamic_batch, k_seqlen, is_quant_flag = is_quant_flag, is_quant_offiline = 1)
        attention_out = torch.zeros_like(self.q).to(outDtype)
        attention_out[:] = 0.1


data_generator = TestPagedAttentionDataGenerator()
# data_generator.test_pa_quant_case_normal_mask()
data_generator.test_pa_quantoffline_case_normal_mask()

(q, k, v, block_table, context_lens, mask, kDescale, vDescale, pScale) = (
    data_generator.q,
    data_generator.key_cache,
    data_generator.value_cache,
    data_generator.block_tables,
    data_generator.contex_lens,
    data_generator.alib_mask,
    data_generator.de_scale1_fp32,
    data_generator.de_scale2_fp32,
    data_generator.scale
)

data = [q, k, v, torch.from_numpy(block_table), torch.from_numpy(context_lens), mask, kDescale, vDescale, pScale]
in_tensors = [tensor.npu() for tensor in data]
_ = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in in_tensors]

OP_NAME = "PagedAttentionOperation"
PARAM = json.dumps({"headNum":32, "qkScale": (1 / 128 ** 0.5), "kvHeadNum": 2, "maskType": 1, "compressType": 0, "quantType": 2, "outDataType": 1})

RUN_PARAM = json.dumps({"contextLens": data_generator.contex_lens.tolist()})

class TestPagedAttentionAttentionOperationCompressHead(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [data_generator.golden_out]

    def golden_compare(self, out_tensor, golden_out_tensor):
        ratios = [0.001, 0.001, 0.005, 0.005]
        return data_generator.compare_output_data(out_tensor.cpu(), golden_out_tensor.cpu(), ratios)

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6], in_tensors[7], in_tensors[8]])


if __name__ == '__main__':
    unittest.main()
 