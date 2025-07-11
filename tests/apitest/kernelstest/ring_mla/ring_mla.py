#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import logging
import unittest
import math
import numpy as np
import sys, os
import op_test
import torch
sys.path.append('/opt/cl/ascend-op-common-lib/tests/pythontest')
from enum import Enum
import random
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
#将父级的父级目录添加到 sys.path
sys.path.append(parent_dir)
from precision_calcu import *

class ScaleType(Enum):
    SCALE_TOR = 0
    SCALE_LOGN = 1
    SCALE_LOGN_FP32 = 2
np.random.seed(123)
MASK_TYPE_NO_MASK = 0
MASK_TYPE_NO_HEAD = 1
MASK_TYPE_NO_BATCH = 2
MASK_TYPE_ALIBI_WITH_BATCH = 3
MASK_TYPE_ALIBI_NO_BATCH = 4
MASK_TYPE_NO_HEAD_DECODER = 5
MASK_TYPE_SWA = 6
MASK_TYPE_SWA_DECODER = 7
MASK_TYPE_ALIBI_WITH_PREFIX_BATCH = 8
MASK_TYPE_NO_BATCH_WITH_PREFIX = 9
MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX = 10
MASK_TYPE_RAZOR_FUSION = 11


def softmax1(
        qk_result,
        is_first,
        gm
):
    sim = qk_result
    lm = torch.max(sim, dim=-1, keepdim=True)[0]

    if is_first:
        hm = lm
        dm = torch.zeros_like(hm)
    else:
        hm = torch.max(gm, lm)
        dm = gm - hm

    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.exp(sim_sub.to(torch.float32))

    row_sum = torch.sum(sim_sub, dim=-1, keepdim=True)
    return sim_sub, row_sum, dm, gm


def qkMM1(
        query,
        key
):
    result = None
    qk_k = key.shape[1]
    for qk_k_split in range(0, qk_k, 128):
        sub_k = 128
        if qk_k_split == 512:
            sub_k = 64
        query_k = query[:, :, qk_k_split: qk_k_split + sub_k]
        key_k = key[:, qk_k_split: qk_k_split + sub_k, :]
        result_split = torch.matmul(query_k.to(torch.float32), key_k.to(torch.float32))
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result


def ref_flash_attention(
        query,
        key,
        value,
        scale,
        alibi_bias,
        mask_data_type=torch.bfloat16,
        query_rope=None,
        key_rope=None,
        context_len=0,
        mask_type = 0,
        data_type=torch.float16
):
    if not torch.is_tensor(query):
        query = torch.from_numpy(query)
    if not torch.is_tensor(key):
        key = torch.from_numpy(key)
    if not torch.is_tensor(value):
        value = torch.from_numpy(value)
    if not torch.is_tensor(alibi_bias):
        alibi_bias = torch.from_numpy(alibi_bias)

    query = torch.permute(query, (1, 0, 2))
    key = torch.permute(key, (1, 2, 0))
    value = torch.permute(value, (1, 0, 2))
    context_size = 128
    group_num = query.shape[0] // key.shape[0]
    if group_num != 1:
        key = key.repeat_interleave(group_num, dim=0)
        value = value.repeat_interleave(group_num, dim=0)
    gl = None
    gl_high = None
    go = None
    go_high = None

    for kv_start in range(0, context_len - 1, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = alibi_bias[:, :, kv_start: kv_start + sub_len]
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key)
        qk_result_high = qkMM1(query.to(torch.float32), sub_key.to(torch.float32))
        qk_result = qk_result.to(data_type) * scale
        qk_result_high = qk_result_high * scale

        if mask_type != 0:
            qk_result += sub_mask
            qk_result_high += sub_mask.to(torch.float32)
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm)
        if kv_start == 0:
            gm_high = None
        p_result_high, row_sum_high, dm_high, gm_high = softmax1(qk_result_high, kv_start == 0, gm_high)
        lo = torch.matmul(p_result.to(torch.float32), sub_value.to(torch.float32))
        # lo = lo.to(data_type)
        lo_high = torch.matmul(p_result_high, sub_value.to(torch.float32))
        # lo = lo.numpy()
        # lo_high = lo_high.numpy()
        if kv_start == 0:
            gl = row_sum
            gl_high = row_sum_high
            go = lo
            go_high = lo_high
        else:
            dm = torch.exp(dm)
            dm_high = torch.exp(dm_high)
            gl = gl * dm
            gl = gl + row_sum

            go = go * dm
            go = go + lo

            gl_high = gl_high * dm_high
            gl_high = gl_high + row_sum_high

            go_high = go_high * dm_high
            go_high = go_high + lo_high
    go = go / gl
    go_high = go_high / gl_high
    go = torch.permute(go, (1, 0, 2))
    go_high = torch.permute(go_high, (1, 0, 2))
    return go, go_high, gl, gm



class TestMLAPrefill(op_test.OpTest):

    def close_pack(self, in_data, seq_len):
        kv = in_data.numpy()
        dim1len = np.size(kv, -2)
        if max(seq_len) > dim1len:
            return None
        kv = kv.reshape(np.prod(kv.shape[0:-1]), kv.shape[-1])
        c_offset = 0
        s_offset = 0
        for i, len in enumerate(seq_len):
            kv[c_offset:c_offset + seq_len[i]][:] = kv[s_offset:s_offset + seq_len[i]][:]
            c_offset += seq_len[i]
            s_offset += dim1len
        return torch.from_numpy(kv[0:sum(seq_len)][:])


    def set_data_params(self, dynamic_batch=False, batch_state=None, window_size=0, cache_type=0,
                        is_mask=True, is_decoder=False, is_alibi=False, is_razor_fusion = False, alibi_dim=4,
                        batch = 1, kv_head = 1, heads = 1, embeddim = 128, embeddimv = 0, max_seq = 2048,
                        kv_seqLen = [], is_clamp = 0, clamp_min = 0, preTokens = 0, nextTokens = 0,
                        tileQ = 0, tileKv = 0, razorLen = 0, baseM = 0, textQLen = 0, textKvLen = 0,
                        is_splitm = False,
                        clamp_max = 0, data_type = torch.float16, op_type = 0, mask_type = 0,
                        no_cache = False, long_seq = False, is_triu_mask = False, is_multi_layer = False,
                        is_sqrt = False, left_align = False, scaleType = ScaleType.SCALE_TOR.value, fav3 = False,
                        tor = 1, bnsd = False, is_compress = False, q_seqlens=None, num_blocks=None,
                        block_size=None,lse=None, last_o=None, isring=0):
        self.dynamic_batch = dynamic_batch
        self.batch_state = batch_state
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.is_alibi = is_alibi
        self.preTokens = preTokens
        self.nextTokens = nextTokens
        self.tileQ = tileQ
        self.tileKv = tileKv
        self.razorLen = razorLen
        self.baseM = baseM
        self.textQLen = textQLen
        self.textKvLen = textKvLen
        self.is_razor_fusion = is_razor_fusion
        self.alibi_dim = alibi_dim
        self.batch = batch
        self.kv_head = kv_head
        self.heads = heads
        self.embeddim = embeddim
        self.embeddimv = embeddimv
        self.max_seq = max_seq
        self.kv_seqLen = kv_seqLen
        self.dynamic_batch = dynamic_batch
        self.is_clamp = is_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.data_type = data_type
        self.no_cache = no_cache
        self.long_seq = long_seq
        self.mask_type = mask_type
        self.is_triu_mask = is_triu_mask
        self.is_multi_layer = is_multi_layer
        self.is_sqrt = is_sqrt
        self.left_align = left_align
        self.fav3 = fav3
        self.scaleType = scaleType
        self.tor = tor
        self.is_int8_flag = False
        self.online = False
        self.bnsd = bnsd
        self.window_size = window_size
        self.is_compress = is_compress
        self.cache_type = cache_type
        self.q_seqlens = q_seqlens if q_seqlens is not None else kv_seqLen
        self.input_lse = lse
        self.last_o = last_o
        self.isring = isring

        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if is_decoder:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        else:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, self.q_seqlens)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        # gen intensor for fa kernel
        if is_multi_layer:
            self.layer_id = torch.from_numpy(np.array([1], dtype=np.int32)).to(torch.int32)
        else:
            self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, heads * self.embeddim)))

        self.q = q.to(data_type)
        if num_blocks is None:
            self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
            self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)
            if is_splitm:
                maxKvSeqlen = max(self.kv_seqlen)
                self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, maxKvSeqlen, kv_head * self.embeddim))).to(data_type)
                self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, maxKvSeqlen, kv_head * self.embeddimv))).to(data_type)
        else:
            # kv cache shape: (num_blocks, block_size, num_heads, head_size)
            self.k_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)
            self.v_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)

            batch = len(kv_seqLen)
            max_context_len = max(kv_seqLen)
            max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
            block_tables = []   # (batch, max_num_blocks_per_seq)
            offset = 0
            for i in range(batch):
                num_blocks_cur_seq = (kv_seqLen[i] + block_size - 1) // block_size
                # padding block table with 0
                block_table = [
                    random.randint(0, num_blocks-1) if j < num_blocks_cur_seq else 0 for j in range(max_num_blocks_per_seq)
                ]
                offset += num_blocks_cur_seq
                block_tables.append(block_table)
            self.block_tables = torch.from_numpy(np.array(block_tables)).to(torch.int32)
            self.k = torch.stack([self.k_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.v = torch.stack([self.v_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.k = self.k.reshape(1, batch, max_context_len, kv_head * self.embeddim)
            self.v = self.v.reshape(1, batch, max_context_len, kv_head * self.embeddim)

        if self.fav3:
            self.is_int8_flag = True
            self.q_scale, self.q_offset, _ = self.quant_per_head(self.q, heads, embeddim, (self.q_ntokens, heads * self.embeddim))
            self.k_scale, self.k_offset, _ = self.quant_per_head(self.k, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.v_scale, self.v_offset, _ = self.quant_per_head(self.v, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.k_scale = (self.k_scale.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.k_offset= (self.k_offset.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_scale = (self.v_scale.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_offset= (self.v_offset.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.offline_scale = torch.from_numpy(np.random.uniform(1 / 127, 3 / 127, size=(heads))).to(torch.float32)

            self.q_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, heads * self.embeddim))).to(torch.int8)
            self.k_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(torch.int8)
            self.v_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(torch.int8)

        self.gen_mask(batch, heads, data_type, mask_type, window_size, is_compress, cache_type)
        logging.info("**********data gen shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"layer_id shape: {self.layer_id.shape}")
        logging.info(f"mask shape: {self.mask.shape}")

    def quant_per_head(self, data, heads, embeddim, shape):
        temp = data.view(-1, heads, self.embeddim).to(torch.float32)
        scale = torch.stack([self.fav3_quant(temp[:, i, :], data_min = -1, data_max = 1, symmetric = True)[0] for i in range(heads)])
        offset = torch.stack([self.fav3_quant(temp[:, i, :], data_min = -1, data_max = 1, symmetric = True)[1] for i in range(heads)])
        int8_data = torch.zeros_like(temp)
        for i in range(heads):
            int8_data[:, i, :] = ((temp[:, i, :] / scale[i]).round_() + offset[i])
        int8_data = int8_data.view(shape).to(torch.int8)
        return scale, offset, int8_data

    def fav3_quant(self, data, data_min = 0, data_max = 0, symmetric = False, bit = 8):
        n = 2 ** (bit - 1)
        if symmetric:
            quant_min, quant_max = -(n - 1), (n - 1)
        else:
            quant_min, quant_max = -n, (n - 1)
        span = quant_max - quant_min
        if data_min == data_max:
            data_max = data.max().item()
            data_min = data.min().item()
        if symmetric:
            scale = max(data_max, -data_min) / (float(span) / 2)
            offset = 0
        else:
            scale = (data_max - data_min) / float(span)
            offset = (data_min * quant_min + data_max * quant_max) / (data_min - data_max)
        # 量化公式：x / scale + offset
        return torch.tensor(float(scale), dtype = torch.float), torch.tensor(int(offset), dtype = torch.float)

    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** ( -4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        return slopes

    def get_alibi_bias(self, n_heads, max_seqlen):
        if not self.left_align:
            self.bias = torch.arange(max_seqlen)
            self.bias = self.bias[None, :] - self.bias[:, None]
            if (self.is_sqrt):
                self.bias = torch.sqrt(torch.abs(self.bias)) * torch.sign(self.bias)
            bias = torch.empty(
                n_heads,
                max_seqlen,
                max_seqlen
            )[:, :max_seqlen, :max_seqlen].copy_(self.bias)
            self.alibi_slopes = self.get_alibi_slopes(n_heads)
        else:
            self.bias = torch.arange(max_seqlen, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(n_heads, max_seqlen, -1)
            self.alibi_slopes = torch.Tensor(self.get_interleave(n_heads))
            bias = self.bias
        bias = bias * self.alibi_slopes[:, None, None]
        return bias

    def get_interleave(self, n, alibi_bias_max=8.0):
        def get_interleave_power_of_2(n, alibi_bias_max):
            if n == 0:
                return 0
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_interleave_power_of_2(n, alibi_bias_max)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
                self.get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def gen_swa_cmp(self, max_seq, window_size):
        swa_mask = np.ones(shape=(1, 512, 512)) * self.pre_mask_coff
        pp_n = 128 if self.embeddim <= 128 else 64
        if window_size <= pp_n * 3:
            true_size = window_size
        else:
            if window_size % pp_n == 0:
                true_size = pp_n * 3
            else:
                true_size = pp_n * 2 + window_size % pp_n
        triu_mask = np.triu(swa_mask, 1)
        tril_mask = np.tril(swa_mask, -true_size)
        swa_mask = triu_mask + tril_mask
        swa_mask = torch.from_numpy(swa_mask).to(torch.float32)
        return swa_mask

    def gen_razor_fusion_mask(self, razorLen, tileQ, tileKv, textQLen, textKvLen, preTokens, nextTokens, baseM):
        np.set_printoptions(threshold=np.inf)

        mask_sizeQ = razorLen * tileQ + textQLen
        mask_sizeK = razorLen * tileKv + textKvLen
        mask = np.zeros((mask_sizeQ, mask_sizeK), dtype=int)
        preTokensBlock = preTokens // baseM
        nextTokensBlock = nextTokens // baseM
        idx = razorLen // baseM * baseM
        mask[:, int(idx) : int(razorLen)] = 0
        mask[int(idx) : int(razorLen), :] = 0
        for i in range((razorLen + baseM - 1) // baseM):
            start =  i - preTokensBlock + 1 if i >= preTokensBlock else 0
            end =  i + nextTokensBlock if i < preTokensBlock else start + preTokensBlock + nextTokensBlock - 1
            end = (razorLen + baseM - 1) // baseM if end > (razorLen + baseM - 1) // baseM else end
            for j in range(start, end):
                mask[i * baseM : (i + 1) * baseM, j * baseM : (j + 1) * baseM] = 1
        mask[razorLen :, :] = 0
        mask[:, razorLen :] = 0
        for i in range(tileQ):
            for j in range(tileKv):
                mask[i * razorLen : (i + 1) * razorLen, j * razorLen : (j + 1) * razorLen] = mask[0 : razorLen, 0 : razorLen]

        mask[razorLen * tileQ : , :] = 1
        mask[: , razorLen * tileKv :] = 1
        mask = mask[None, None, :]
        mask = 1 - mask
        return mask * -10000

    def gen_swa_mask(self, max_seq, window_size, pre_mask_coff, cache_type=0):
        swa_mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        if window_size < max_seq and self.is_decoder:
            if cache_type == 1:
                for idx, kvseqlen in enumerate(self.kv_seqLen):
                    swa_mask[idx, :, :window_size] = 0
            else:
                for idx, kvseqlen in enumerate(self.kv_seqLen):
                    swa_mask[idx, :, kvseqlen - window_size: kvseqlen] = 0
        elif window_size < max_seq or self.is_compress:
            triu_mask = np.triu(swa_mask, 1)
            tril_mask = np.tril(swa_mask, -window_size)
            swa_mask = triu_mask + tril_mask
        else:
            swa_mask = np.triu(swa_mask, 1)
        return swa_mask

    def gen_mask(self, batch, heads, data_type, mask_type, window_size, is_compress, cache_type=0):
        import random
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH : ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_WITH_PREFIX_BATCH : ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, kv_s-q_s:kv_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_NO_HEAD : ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER : ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH_WITH_PREFIX : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_SWA : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_SWA_DECODER : ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            # 不加mask
            MASK_TYPE_RAZOR_FUSION : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:q_s, :kv_s]))),
            MASK_TYPE_NO_MASK : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: 0))
        }
        # kernel中mask的系数
        if data_type == torch.float16:
            post_mask_coff = 1
            pre_mask_coff = -10000.0
        elif data_type == torch.bfloat16 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = -float("inf")
        elif data_type == torch.float32 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = 1
        else:
            post_mask_coff = -3e38
            pre_mask_coff = 1
        if data_type == torch.float16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi or self.long_seq:
                select_zero = False
            else:
                select_zero = True
        elif data_type == torch.bfloat16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi:
                select_zero = False
            elif self.dynamic_batch or self.is_decoder:
                select_zero = True
            else:
                select_zero = False
        else:
            if self.is_alibi or self.is_decoder:
                select_zero = True
            else:
                select_zero = False
        if self.is_triu_mask:
            select_zero = False

        self.mask_info = mask_type_dict[mask_type]
        mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        mask = np.triu(mask, 1)
        zero_indice = random.choices(range(self.max_seq), k = 300)
        if self.window_size > 0:
            mask = self.gen_swa_mask(self.max_seq, window_size, pre_mask_coff, cache_type)
        if self.is_alibi:
            self.alibi_bias = self.get_alibi_bias(heads, self.max_seq)
            mask += self.alibi_bias.numpy()
        if select_zero:
            mask.flat[zero_indice] = 0
        if self.is_razor_fusion:
            mask = self.gen_razor_fusion_mask(self.razorLen, self.tileQ, self.tileKv, self.textQLen, self.textKvLen,
                                              self.preTokens, self.nextTokens, self.baseM)
            post_mask_coff = 1
        self.mask = torch.from_numpy(mask).to(torch.float32)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff

    def quantize_tensor_symmetric(self, x, prev_max_abs_vals=None, num_bits=8):
        if x.dtype != torch.float:
            x = x.to(torch.float)

        quant_min = -2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1

        current_max_abs_vals = x.abs().max(dim=1).values
        if prev_max_abs_vals is not None:
            max_abs_vals = torch.max(prev_max_abs_vals, current_max_abs_vals)
        else:
            max_abs_vals = current_max_abs_vals
        scales = max_abs_vals / (quant_max)
        x_q = torch.clamp(torch.round(x / scales.unsqueeze(1)), quant_min, quant_max)
        x_q = torch.round(x_q)
        x_q = x_q.to(torch.int8)
        return x_q, scales, max_abs_vals

    def dequantize_tensor(self, x_q, scales, value):
        x_deq = x_q.to(torch.float32)
        scales = scales.unsqueeze(1)
        x_deq = x_deq * value
        x_deq = x_deq * scales
        return x_deq

    def online_softmax(self, s_qk, q_s, v_slice, heads, kv_head, embed, online, dtype):
        ans = None
        group_num = heads // kv_head
        for head_idx in range(heads):
            s_head_idx = s_qk[head_idx]
            O = torch.zeros((q_s, embed)).to(dtype)
            Br = q_s
            Bc = 128
            self.row_block_size = Br
            self.col_block_size = Bc
            d = embed
            V_mat = v_slice[head_idx // group_num]
            Tr = q_s // Br
            Tc = q_s // Bc

            d = embed
            Tr = q_s // Br
            Tc = q_s // Bc

            start_row_idx = 0
            start_col_idx = 0

            for i in range(Tr):

                Oi = torch.zeros((Br, d)).to(dtype)  # shape Br x d
                li = torch.zeros((Br, 1)).to(dtype)  # shape Br x 1
                mi = torch.full((Br, 1), -torch.inf).to(dtype)  # shape Br x 1
                pp_max_num = None

                for j in range(Tc):

                    Sij = s_head_idx[i *  Br : (i + 1) * Br, start_col_idx + j * Bc : start_col_idx + (j + 1) * Bc].to(dtype)

                    Vj = V_mat[start_col_idx + j * Bc : start_col_idx + (j + 1) * Bc, :]

                    mi_new = torch.max(
                        torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1
                    ).values[:, None].to(dtype)
                    Pij_hat = torch.exp((Sij - mi_new).to(torch.float32))
                    Pij_hat = Pij_hat.to(dtype)
                    li = torch.exp((mi - mi_new).to(torch.float32)).to(dtype) * li + torch.sum(Pij_hat, dim=1)[:, None]
                    if self.is_int8_flag:
                        if online:
                            x_q, scales, pp_max_num = self.quantize_tensor_symmetric(Pij_hat, pp_max_num)
                            if pp_max_num == None:
                                pp_max_num = pp_max_num
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + self.dequantize_tensor(pv, scales, self.v_scale[head_idx]).to(dtype)
                        else:
                            x_q = Pij_hat / self.offline_scale[head_idx]
                            x_q = torch.round(x_q.to(torch.float32))
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            pv = pv.to(torch.float32)
                            value = self.v_scale[head_idx] * self.offline_scale[head_idx]
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + (pv * value).to(dtype)
                    else:
                        Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + Pij_hat @ Vj.to(dtype)

                    mi = mi_new

                if (q_s % Bc != 0):
                    Bc = q_s % Bc
                    start_row_idx = (q_s // self.row_block_size) * self.row_block_size
                    start_col_idx = (q_s // self.col_block_size) * self.col_block_size

                    Sij = s_head_idx[i *  Br : (i + 1) * Br, start_col_idx : start_col_idx + Bc].to(dtype)
                    Vj = V_mat[start_col_idx : start_col_idx + Bc, :]
                    mi_new = torch.max(
                        torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1
                    ).values[:, None].to(dtype)
                    Pij_hat = torch.exp((Sij - mi_new).to(torch.float32))
                    Pij_hat = Pij_hat.to(dtype)
                    li = torch.exp((mi - mi_new).to(torch.float32)).to(dtype) * li + torch.sum(Pij_hat, dim=1)[:, None]
                    if self.is_int8_flag:
                        if online:
                            x_q, scales, pp_max_num = self.quantize_tensor_symmetric(Pij_hat, pp_max_num)
                            if pp_max_num == None:
                                pp_max_num = pp_max_num
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + self.dequantize_tensor(pv, scales, self.v_scale[head_idx]).to(dtype)
                        else:
                            x_q = Pij_hat / self.offline_scale[head_idx]
                            x_q = torch.round(x_q.to(torch.float32))
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            pv = pv.to(torch.float32)
                            value = self.v_scale[head_idx] * self.offline_scale[head_idx]
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + (pv * value).to(dtype)
                    else:
                        Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + Pij_hat @ Vj.to(dtype)
                Oi = Oi / li

                O[i * Br : (i + 1) * Br, :] = Oi

            if ans is None:
                ans = O
            else:
                ans = torch.cat((ans, O), 1)
        return ans

    def update_out(self, out, lse):
        result = None
        result_lse = None
        result_lse_new = None
        tmp_o = None
        temp_input_lse = self.input_lse.reshape(self.q_ntokens, self.heads, 1)
        # 更新o
        out = out.view(self.q_ntokens, self.heads, self.embeddimv)
        temp_o = self.last_o.view(self.q_ntokens, self.heads, self.embeddimv)  # [q_len, head, embeddimv

        temp_o = temp_o.to(torch.float32)
        out  = out.to(torch.float32)
        lse = lse.to(torch.float32)
        temp_input_lse = temp_input_lse.to(torch.float32)

        for batch_idx in range(self.batch):
            q_start_idx = sum(self.q_seqlens[:batch_idx])
            q_len = self.q_seqlens[batch_idx]

            if self.kv_seqLen[batch_idx] == 0:
                batch_out = temp_o[q_start_idx:(q_start_idx+q_len)].view(-1, self.heads * self.embeddimv)
                if result is None:
                    result = batch_out
                else:
                    result = torch.cat((result, batch_out), 0)
                continue

            lse_i = lse[q_start_idx:(q_start_idx+q_len)].permute(1, 0).unsqueeze(0)  # (1, heads, bs)

            lse_old_exp = temp_input_lse[q_start_idx:(q_start_idx+q_len)].permute(2, 1, 0)
            lse_old_exp = torch.exp(lse_old_exp)

            out_i = out[q_start_idx:(q_start_idx+q_len)]
            last_o_i = temp_o[q_start_idx:(q_start_idx+q_len)]

            lse_new_exp = torch.exp(lse_i)  # [1, head, q_len]
            lse_old_exp = lse_old_exp.permute(2, 1, 0).repeat(1, 1, self.embeddimv)  # (q_len, head, embeddimv)
            lse_new_exp = lse_new_exp.permute(2, 1, 0).repeat(1, 1, self.embeddimv)
            fenmu = lse_old_exp + lse_new_exp

            fenzi2 = out_i * lse_new_exp
            fenzi1 = last_o_i * lse_old_exp

            if tmp_o is None:
                tmp_o = fenzi2
            else:
                tmp_o = torch.cat((tmp_o, fenzi2), 0)

            new_out = (fenzi2 + fenzi1) / fenmu
            batch_out = new_out.view(-1, self.heads * self.embeddimv)

            if result is None:
                result = batch_out
            else:
                result = torch.cat((result, batch_out), 0)

        for batch_idx in range(self.batch):
            q_start_idx = sum(self.q_seqlens[:batch_idx])
            q_len = self.q_seqlens[batch_idx]

            if self.kv_seqLen[batch_idx] == 0:
                new_lse_aaa = torch.zeros_like(lse[q_start_idx:(q_start_idx+q_len)]).unsqueeze(-1)
                if result_lse_new is None:
                    result_lse_new = new_lse_aaa
                else:
                    result_lse_new = torch.cat((result_lse_new, new_lse_aaa), 0)
                continue

            lse_i = lse[q_start_idx:(q_start_idx+q_len)].permute(1, 0).unsqueeze(0)  # (1, heads, bs)
            lse_old = temp_input_lse[q_start_idx:(q_start_idx+q_len)].permute(2, 1, 0)  # (1, heads, bs)

            new_lse_aaa = torch.log(torch.exp(lse_old) + torch.exp(lse_i))  # (1, heads, bs)
            new_lse_aaa = new_lse_aaa.permute(2, 1, 0)
            if result_lse_new is None:
                result_lse_new = new_lse_aaa
            else:
                result_lse_new = torch.cat((result_lse_new, new_lse_aaa), 0)

        temp_lse = result_lse_new.transpose(1, 0).squeeze(-1)

        return result, temp_lse

    def gen_out_tensor(self, online=False):
        print("-----------------------start-----------------------------")
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        embedv = self.embeddimv
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        is_razor_fusion = self.is_razor_fusion
        q = self.q
        k = self.k
        v = self.v
        if self.fav3:
            q = self.q_int8
            k = self.k_int8
            v = self.v_int8
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        ans_concat = None
        ans_concat_true = None
        out_true = None

        self.encoder_logN = torch.tensor([2.0] * self.max_seq).to(torch.float32)
        self.encoder_logN.uniform_(1, 2)
        self.decoder_logN = torch.tensor([2.0] * batch).to(torch.float32)
        self.decoder_logN.uniform_(1, 2)
        self.new_lse = None
        self.new_lse_height = None

        out_height = None
        for idx in range(batch):
            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue
            if dynamic_batch and batch_state[idx] == 0:
                output = torch.zeros([heads, q_s, embedv])
                output = torch.permute(output, (1, 0, 2))
                if out is None:
                    out = output
                    if not self.fav3:
                        out_true = output
                else:
                    out = torch.cat((out, output), 0)
                    if not self.fav3:
                        out_true = torch.cat((out_true, output), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            if kv_s == 0:
                o = torch.zeros(size=(q_s, heads, embedv), dtype=self.data_type)
                if out == None:
                    out = o
                else:
                    out = torch.cat((out, o), 0)
                if out_height == None:
                    out_height = o
                else:
                    out_height = torch.cat((out_height, o), 0)
                q_offset += q_s
                lse = torch.zeros(size=(self.heads, q_s, 1), dtype = torch.float32)
                if self.new_lse is None:
                    self.new_lse = lse
                else:
                    self.new_lse = np.concatenate((self.new_lse, lse), axis=1)   # shape is (heads, bs)
                continue
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice_ori = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice_ori, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice_ori = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice_ori, (1, 2, 0))   # get K^T
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice_ori = v_slice.view(kv_s, kv_head, embedv)
            v_slice = torch.permute(v_slice_ori, (1, 0, 2))

            temp_mask = self.mask_info[1](self.mask, idx, q_s, kv_s) * self.post_mask_coff
            if  self.mask_type != MASK_TYPE_NO_MASK:
                temp_mask = temp_mask.repeat(self.heads, 1, 1)[:, :q_s, :]
            else:
                temp_mask = self.mask.repeat(self.heads, 1, 1)[:, :q_s, :]

            context_len = k_slice_ori.shape[0]
            new_out, new_out_height, gl, gm = ref_flash_attention(q_slice_ori, k_slice_ori, v_slice_ori, self.tor, temp_mask, context_len=context_len, mask_type=self.mask_type, data_type=self.data_type)

            if out == None:
                out = new_out
            else:
                out = torch.cat((out, new_out), 0)

            if out_height == None:
                out_height = new_out_height
            else:
                out_height = torch.cat((out_height, new_out_height), 0)

            lse = torch.log(gl) + gm

            if self.new_lse is None:
                self.new_lse = lse
            else:
                self.new_lse = torch.cat((self.new_lse, lse), dim=1) # shape is (heads, bs)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq

        # golden data
        self.new_lse = torch.squeeze(self.new_lse, dim=-1)
        self.new_lse = self.new_lse.permute(1, 0).contiguous()
        self.out_lse = self.new_lse.reshape(self.q_ntokens * self.heads, 1)  # (bs * head, 1)

        if self.is_int8_flag:
            ans_concat = ans_concat.view(q_ntokens, heads * embedv)
            ans_concat_true = ans_concat_true.view(q_ntokens, heads * embedv)
            self.golden_out = ans_concat
            self.golden_out_true = ans_concat_true
        else:
            if not self.isring:
                out = out.view(q_ntokens, heads * embedv)
                self.golden_out_o = out.to(self.data_type)
                self.golden_out = out.to(self.data_type)
                out_true = out_height.view(q_ntokens, heads * embedv)
                self.golden_out_true = out_true.to(torch.float32)
                self.new_lse = self.new_lse.transpose(1, 0)
            else:
                out = out.view(q_ntokens, heads * embedv)
                new_out, new_lse = self.update_out(out, self.new_lse)
                self.golden_out = new_out.to(self.data_type)
                self.new_lse_low = new_lse

                out_true = out_height.view(q_ntokens, heads * embedv)
                new_out, new_lse_height = self.update_out(out_true, self.new_lse)
                self.golden_out_true = new_out.to(torch.float32)
                self.new_lse_height = new_lse_height

        if self.no_cache:
            self.k = self.close_pack(self.k.to(torch.float32), kv_seqlen).to(self.data_type)
            self.v = self.close_pack(self.v.to(torch.float32), kv_seqlen).to(self.data_type)
            if self.fav3:
                self.k_int8 = self.close_pack(self.k_int8.to(torch.float32), kv_seqlen).to(torch.int8)
                self.v_int8 = self.close_pack(self.v_int8.to(torch.float32), kv_seqlen).to(torch.int8)
        if self.long_seq:
            self.max_seq = 512
            self.gen_mask(self.batch, self.heads, self.data_type, self.mask_type, 0, False, 0)

        self.q_split1, self.q_split2 = q[:, :128], q[:, 128:192]
        for i in range(1, self.heads):
            self.q_split1 = torch.cat([self.q_split1, q[:,i*192:i*192+128]], dim = 1)
            self.q_split2 = torch.cat([self.q_split2, q[:,i*192+128:(i+1)*192]], dim = 1)

        self.k_split1, self.k_split2 = k[:,:,:, :128], k[:,:,:, 128:192]
        for i in range(1, self.kv_head):
            self.k_split1 = torch.cat([self.k_split1,  k[:,:,:, i*192:i*192+128]], dim = 3)
            self.k_split2 = torch.cat([self.k_split2, k[:,:,:,i*192+128:(i+1)*192]], dim = 3)

    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.flatten().to(torch.float32)

        out = out.flatten().to(torch.float32)
        out_len = out.shape[0]

        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        logging.info(f"maxDiff {max_diff}")
        logging.info("1/1000 Accuracy is %f",  1 - float(error_count) / out_len)
        logging.info("5/1000 Accuracy is %f",  1 - float(strict_error_count) / out_len)
        if self.data_type == torch.bfloat16:
            logging.debug("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[2])
        else:
            logging.debug("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[0])
        calc_times = self.heads * self.max_seq + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else :
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res
        elif self.data_type == torch.float16:
            if calc_times < 2048:
                error = 2**(-8)
            else :
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res
        else :
            if calc_times < 2048:
                error = 2**(-11)
            elif calc_times >= 2048 and calc_times < 16384:
                error = 2**(-10)
            else:
                error = 2**(-14)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res

    def group_mm_torch(self, heads, group_num, A, B, dtype=torch.float32):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(dtype), B[i:(i + 1), :, :].to(dtype))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out)
        if self.isring == 0:
            lse_result = self.new_lse
        else:
            lse_result = self.new_lse_height
        return [golden_out, lse_result]

    def golden_compare(self, out_tensors, golden_tensors):
        result_single = self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.005, 0.005])
        result_single_lse = self.compare_output_data(out_tensors[1].half(), golden_tensors[1].half(), [0.001, 0.001, 0.005, 0.005])
        if self.is_int8_flag:
            result_double = compare_cv(self.golden_out_true, golden_tensors[0], out_tensors[0])
            return (result_double or result_single)
        else:
            result_double = compare_cv(self.golden_out_true, golden_tensors[0], out_tensors[0])
            return (result_double or result_single)
        return True
    def transpose_kv_shape(self, kv_seqLen, kv_head):
        kv_ntokens = sum(kv_seqLen)
        new_k = None
        new_k_rope = None
        new_v = None
        for i, kv_len in enumerate(kv_seqLen):
            if kv_len == 0:
                continue
            if new_k == None:
                new_k = self.k_split1[0][i][:kv_len][:]
            else:
                new_k = torch.cat((new_k, self.k_split1[0][i][:kv_len][:]), 0)
            if new_k_rope == None:
                new_k_rope = self.k_split2[0][i][:kv_len][:]
            else:
                new_k_rope = torch.cat((new_k_rope, self.k_split2[0][i][:kv_len][:]), 0)
            if new_v == None:
                new_v = self.v[0][i][:kv_len][:]
            else:
                new_v = torch.cat((new_v, self.v[0][i][:kv_len][:]))
        new_k = new_k.reshape(kv_ntokens, kv_head, 128)
        new_k_rope = new_k_rope.reshape(kv_ntokens, kv_head, 64)
        new_v = new_v.reshape(kv_ntokens, kv_head, 128)
        return new_k, new_k_rope, new_v


    @op_test.only_910b
    def test_flash_attention_mla_fp16_mask(self):
        batch = 2
        kv_head = 16      # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16        # llama7b  hidden_size 4096
        embeddim = 192
        embeddimV = 128
        max_seq = 200
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [200] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0

        isring = 1
        shape_out_1 = (sum(kv_seqLen), heads, embeddimV)  # embeddimV  sum(q_seq), head*ebeddimv
        shape_out_2 = (sum(kv_seqLen), heads)
        data_type = torch.float16

        if isring:
            old_out = torch.rand(shape_out_1).to(data_type)
            old_lse = (torch.rand(shape_out_2) * 10).to(torch.float32)
        else:
            old_out = torch.zeros(shape_out_1, dtype=data_type)
            old_lse = torch.zeros(shape_out_2, dtype=torch.float32)
        output_lse = torch.zeros(heads, (sum(kv_seqLen)), dtype=torch.float32)

        OP_NAME = "RINGMLAOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor, "isTriuMask": 1, "maskType": 1, "kvHead": heads, "isRing":isring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 15)
        self.set_output_formats([self.format_nd] * 2)


        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_triu_mask = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor, long_seq = True,
                             lse=old_lse, last_o=old_out, isring=isring)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"layer_id shape: {self.layer_id.shape}")
        logging.info(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))

        self.mask = self.mask.view(512, 512).to(data_type)

        old_lse = old_lse.transpose(1, 0)
        self.execute([self.q_split1, self.q_split2, self.k_split1, self.k_split2, self.v, self.mask.to(data_type),
                      torch.tensor([], dtype=torch.float),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float),
                      old_out,old_lse],
                     [torch.tensor(attention_out, dtype=data_type), output_lse])


    @op_test.only_910b
    def test_flash_attention_mla_fp16_nomask(self):
        batch = 2
        kv_head = 16      # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16        # llama7b  hidden_size 4096
        embeddim = 192
        embeddimV = 128
        max_seq = 100
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [100] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0

        isring = 1
        shape_out_1 = (sum(kv_seqLen), heads, embeddimV)  # embeddimV  sum(q_seq), head*ebeddimv
        shape_out_2 = (sum(kv_seqLen), heads)
        data_type = torch.float16

        if isring:
            old_out = torch.rand(shape_out_1).to(data_type)
            old_lse = (torch.rand(shape_out_2) * 10).to(torch.float32)
        else:
            old_out = torch.zeros(shape_out_1, dtype=data_type)
            old_lse = torch.zeros(shape_out_2, dtype=torch.float32)
        output_lse = torch.zeros(heads, (sum(kv_seqLen)), dtype=torch.float32)

        OP_NAME = "RINGMLAOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor, "isTriuMask": 0, "maskType": 0, "kvHead": heads, "isRing":isring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 15)
        self.set_output_formats([self.format_nd] * 2)


        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_triu_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, tor = tor, long_seq = False,
                             lse=old_lse, last_o=old_out, isring=isring,)
        self.gen_out_tensor()
        logging.info(f"mask shape: {self.mask.shape}")
        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"layer_id shape: {self.layer_id.shape}")
        logging.info(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        old_lse = old_lse.transpose(1, 0)
        self.execute([self.q_split1, self.q_split2, self.k_split1, self.k_split2, self.v, torch.tensor([], dtype=data_type),
                      torch.tensor([], dtype=torch.float),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float),
                      old_out,old_lse],
                     [torch.tensor(attention_out, dtype=data_type), output_lse])

    @op_test.only_910b
    def test_flash_attention_mla_bf16_512_nomask(self):
        batch = 2
        kv_head = 16      # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16        # llama7b  hidden_size 4096
        embeddim = 192
        embeddimV = 128
        max_seq = 200
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [200] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0

        isring = 1
        shape_out_1 = (sum(kv_seqLen), heads, embeddimV)  # embeddimV  sum(q_seq), head*ebeddimv
        shape_out_2 = (sum(kv_seqLen), heads)
        data_type = torch.bfloat16

        if isring:
            old_out = torch.rand(shape_out_1).to(data_type)
            old_lse = (torch.rand(shape_out_2) * 10).to(torch.float32)
        else:
            old_out = torch.zeros(shape_out_1, dtype=data_type)
            old_lse = torch.zeros(shape_out_2, dtype=torch.float32)
        output_lse = torch.zeros(heads, (sum(kv_seqLen)), dtype=torch.float32)

        OP_NAME = "RINGMLAOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor, "isTriuMask": 0, "maskType": 0, "kvHead": heads, "isRing":isring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 15)
        self.set_output_formats([self.format_nd] * 2)


        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_triu_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, tor = tor, long_seq = False,
                             lse=old_lse, last_o=old_out, isring=isring,)
        self.gen_out_tensor()
        logging.info(f"mask shape: {self.mask.shape}")
        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"layer_id shape: {self.layer_id.shape}")
        logging.info(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        old_lse = old_lse.transpose(1, 0)
        self.execute([self.q_split1, self.q_split2, self.k_split1, self.k_split2, self.v, torch.tensor([], dtype=data_type),
                      torch.tensor([], dtype=torch.float),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float),
                      old_out,old_lse],
                     [torch.tensor(attention_out, dtype=data_type), output_lse])#

    @op_test.only_910b
    def test_flash_attention_mla_bf16_512_mask(self):
        batch = 2
        kv_head = 16      # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16        # llama7b  hidden_size 4096
        embeddim = 192
        embeddimV = 128
        max_seq = 200
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [200] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0

        isring = 0
        shape_out_1 = (sum(kv_seqLen), heads, embeddimV)  # embeddimV  sum(q_seq), head*ebeddimv
        shape_out_2 = (sum(kv_seqLen), heads)
        data_type = torch.bfloat16

        if isring:
            old_out = torch.rand(shape_out_1).to(data_type)
            old_lse = (torch.rand(shape_out_2) * 10).to(torch.float32)
        else:
            old_out = torch.zeros(shape_out_1, dtype=data_type)
            old_lse = torch.zeros(shape_out_2, dtype=torch.float32)
        output_lse = torch.zeros(heads, (sum(kv_seqLen)), dtype=torch.float32)

        OP_NAME = "RINGMLAOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor, "isTriuMask": 1, "maskType": 1, "kvHead": heads, "isRing":isring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 15)
        self.set_output_formats([self.format_nd] * 2)


        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_triu_mask = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor, long_seq = True,
                             lse=old_lse, last_o=old_out, isring=isring,)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"layer_id shape: {self.layer_id.shape}")
        logging.info(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))

        self.mask = self.mask.view(512, 512).to(data_type)
        old_lse = old_lse.transpose(1, 0)
        self.execute([self.q_split1, self.q_split2, self.k_split1, self.k_split2, self.v, self.mask.to(data_type),
                      torch.tensor([], dtype=torch.float),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float),
                      old_out,old_lse],
                     [torch.tensor(attention_out, dtype=data_type), output_lse])

    @op_test.only_910b
    def test_flash_attention_mla_fp16_nomask_qnotk(self):
        batch = 2
        kv_head = 16      # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16        # llama7b  hidden_size 4096
        embeddim = 192
        embeddimV = 128
        max_seq = 200
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        q_seqLen = [100] * batch
        kv_seqLen = [200] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0

        isring = 1
        shape_out_1 = (sum(q_seqLen), heads, embeddimV)  # embeddimV  sum(q_seq), head*ebeddimv
        shape_out_2 = (sum(q_seqLen), heads)
        data_type = torch.float16

        if isring:
            old_out = torch.rand(shape_out_1).to(data_type)
            old_lse = (torch.rand(shape_out_2) * 10).to(torch.float32)
        else:
            old_out = torch.zeros(shape_out_1, dtype=data_type)
            old_lse = torch.zeros(shape_out_2, dtype=torch.float32)
        output_lse = torch.zeros(heads, (sum(q_seqLen)), dtype=torch.float32)

        OP_NAME = "RINGMLAOperation"
        OP_PARAM = {"type": 1, "qSeqLen":q_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor, "isTriuMask": 0, "maskType": 0, "kvHead": heads, "isRing":isring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 15)
        self.set_output_formats([self.format_nd] * 2)


        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_triu_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, tor = tor, long_seq = False,q_seqlens=q_seqLen,
                             lse=old_lse, last_o=old_out, isring=isring)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"layer_id shape: {self.layer_id.shape}")
        logging.info(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))

        old_lse = old_lse.transpose(1, 0)
        self.execute([self.q_split1, self.q_split2, self.k_split1, self.k_split2, self.v, torch.tensor([], dtype=torch.float), #self.mask.to(data_type),
                      torch.tensor([], dtype=torch.float),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float),
                      old_out,old_lse],
                     [torch.tensor(attention_out, dtype=data_type), output_lse])

    @op_test.only_910b
    def test_flash_attention_mla_fp16_mask_qnotk(self):
        batch = 2
        kv_head = 16      # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16        # llama7b  hidden_size 4096
        embeddim = 192
        embeddimV = 128
        max_seq = 200
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        q_seqLen = [100] * batch
        kv_seqLen = [200] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0

        isring = 1
        shape_out_1 = (sum(q_seqLen), heads, embeddimV)  # embeddimV  sum(q_seq), head*ebeddimv
        shape_out_2 = (sum(q_seqLen), heads)
        data_type = torch.float16

        if isring:
            old_out = torch.rand(shape_out_1).to(data_type)
            old_lse = (torch.rand(shape_out_2) * 10).to(torch.float32)
        else:
            old_out = torch.zeros(shape_out_1, dtype=data_type)
            old_lse = torch.zeros(shape_out_2, dtype=torch.float32)
        output_lse = torch.zeros(heads, (sum(q_seqLen)), dtype=torch.float32)

        OP_NAME = "RINGMLAOperation"
        OP_PARAM = {"type": 1, "qSeqLen":q_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor, "isTriuMask": 1, "maskType": 1, "kvHead": heads, "isRing":isring}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 15)
        self.set_output_formats([self.format_nd] * 2)


        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_triu_mask = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor, long_seq = True,q_seqlens=q_seqLen,
                             lse=old_lse, last_o=old_out, isring=isring)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"layer_id shape: {self.layer_id.shape}")
        logging.info(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        self.mask = self.mask.view(512, 512).to(data_type)

        old_lse = old_lse.transpose(1, 0)
        self.execute([self.q_split1, self.q_split2, self.k_split1, self.k_split2, self.v, self.mask.to(data_type),
                      torch.tensor([], dtype=torch.float),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                      torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float),
                      old_out,old_lse],
                     [torch.tensor(attention_out, dtype=data_type), output_lse])


if __name__ == '__main__':
    unittest.main()
