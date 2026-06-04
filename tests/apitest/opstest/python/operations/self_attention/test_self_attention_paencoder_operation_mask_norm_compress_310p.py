#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import json
import math
import os
import sys
import unittest
import random
import numpy as np
import torch
import torch_npu

np.random.seed(0)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


def group_matmul(heads, group_num, A, B):
    group_head = heads // group_num
    score = None
    for i in range(group_num):
        group_score = np.matmul(
            A[i * group_head:(i + 1) * group_head, :, :].astype(np.float32),
            B[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    return score


def round_up(x, align):
    if align == 0:
        return -1
    return (x + align - 1) // align * align


def custom_pad(x, pad_dims):
    return torch.nn.functional.pad(x, pad_dims)


def custom_reshape(x, target_shape):
    return x.reshape(target_shape)


def custom_transpose(x, dim1, dim2):
    return x.transpose(dim1, dim2)


def nd_to_nz_2d(in_tensor):
    aux_dims = [0, 0, 0, 0]
    aux_dims[0] = 1
    aux_dims[1] = round_up(in_tensor.size(0), 16)

    pad_dims = [0, 0, 0, 0]
    pad_dims[3] = round_up(in_tensor.size(0), 16) - in_tensor.size(0)

    aux_dims[2] = round_up(in_tensor.size(1), 16) // 16
    aux_dims[3] = 16
    pad_dims[1] = round_up(in_tensor.size(1), 16) - in_tensor.size(1)

    return custom_transpose(
        custom_reshape(custom_pad(in_tensor, pad_dims), aux_dims), 1,
        2).contiguous()


def calc_expect_func(batch, qkv_seqlen, maxSeqlen, heads, kv_head, embed):
    is_mask = True
    src_type = 'float16'
    fp32 = True
    q_seqlen = np.array(qkv_seqlen)
    kv_seqlen = np.array(qkv_seqlen)
    q_ntokens = q_seqlen.sum()
    kv_ntokens = kv_seqlen.sum()

    max_s = maxSeqlen

    q = np.random.uniform(-1.0, 1.0,
                          size=(q_ntokens, heads * embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens,
                                           kv_head * embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens,
                                           kv_head * embed)).astype(np.float16)

    mask = np.zeros((max_s, max_s), dtype=np.float16)
    # 构造严格上三角（不包括对角线）的 mask，赋值为 -inf
    mask[np.triu_indices(max_s, k=1)] = 1
    mask *= -60000
    mask = mask.reshape(1, max_s, max_s)

    q_offset = 0
    k_offset = 0
    v_offset = 0

    out = None

    for idx in range(batch):
        q_s = q_seqlen[idx]
        kv_s = kv_seqlen[idx]
        q_slice = q[q_offset:q_offset + q_s][:]
        q_slice = q_slice.reshape(q_s, heads, embed)
        q_slice = np.transpose(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
        k_slice = k[k_offset:k_offset + kv_s][:]
        k_slice = k_slice.reshape(kv_s, kv_head, embed)
        k_slice = np.transpose(k_slice, (1, 0, 2))
        # get K^T (kv_heads, embed, k_seq)
        k_slice_t = np.transpose(k_slice, (0, 2, 1))
        v_slice = v[v_offset:v_offset + kv_s][:]
        v_slice = v_slice.reshape(kv_s, kv_head, embed)
        v_slice = np.transpose(v_slice, (1, 0, 2))
        score = group_matmul(heads, kv_head, q_slice, k_slice_t)
        tor = np.float16(1.0 / math.sqrt(1.0 * embed))
        score = score * tor
        if is_mask:
            score = score + mask[:, :q_s, :kv_s]
        score_max = np.max(score, axis=-1)
        score = score - score_max.reshape((heads, q_s, 1))
        score_exp = np.exp(score.astype(np.float32))
        if not fp32:
            score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
            p = score_exp.astype(np.float16) / score_sum.reshape(
                (heads, q_s, 1)).astype(np.float16)
            out_sub = group_matmul(heads, kv_head, p, v_slice)
        else:
            score_sum = np.sum(score_exp, axis=-1)
            p = score_exp.astype(np.float16)
            out_sub = group_matmul(heads, kv_head, p, v_slice)
            out_sub = out_sub / \
                score_sum.reshape((heads, q_s, 1)).astype(np.float16)

        out_sub = out_sub.reshape(heads, q_s, embed)
        out_sub = np.transpose(out_sub, (1, 0, 2))
        out_sub = np.ascontiguousarray(out_sub)
        if out is None:
            out = out_sub
        else:
            out = np.concatenate((out, out_sub), 0)

        q_offset += q_s
        k_offset += kv_s
        v_offset += kv_s

    print("==> data generate finished!")

    q = q.astype(src_type).reshape(-1, heads, embed)
    k = k.astype(src_type).reshape(-1, kv_head, embed)
    v = v.astype(src_type).reshape(-1, kv_head, embed)

    mask_2048 = np.zeros((2048, 2048), dtype=np.float16)
    # 构造严格上三角（不包括对角线）的 mask，赋值为 -inf
    mask_2048[np.triu_indices(2048, k=1)] = 1
    mask_2048 *= -60000
    mask_2048 = mask_2048.astype(src_type).reshape(2048, 2048)


    q_len = q_seqlen.astype(np.int32)
    out = out.astype(src_type).reshape(-1, heads, embed)

    ret_data = q, k, v, mask_2048, q_len, out
    return ret_data


if operation_test.get_soc_version() == 'Ascend310P':
    batch = 9
    qkv_seqlen = [70, 140, 1, 167, 5000, 15, 3000, 4048, 7777]
    maxSeqlen = max(qkv_seqlen)
    heads = 16
    kv_head = 2
    embed = 128
    data = calc_expect_func(batch, qkv_seqlen, maxSeqlen, heads, kv_head, embed)
    param_seqlen = data[4].tolist()
    in_tensors = [torch.from_numpy(tensor) for tensor in data]
    in_tensors = [tensor.npu() for tensor in in_tensors]
    mask_nz = nd_to_nz_2d(in_tensors[3])
    mask_nz = torch_npu.npu_format_cast(mask_nz, 29)
    in_tensors[3] = mask_nz
    a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

    OP_NAME = "SelfAttentionOperation"
    PARAM = json.dumps({
        "headNum": heads,
        "qkScale": (1 / float(math.sqrt(embed))),
        "kvHeadNum": kv_head,
        "calcType": 3,
        "maskType": 3,
        "isTriuMask": 1
    })
    RUN_PARAM = json.dumps({"seqLen": param_seqlen})
    print(PARAM, RUN_PARAM)


class TestFlashAttentionPAEncoderOperation310P(operation_test.OperationTest):

    def golden_calc(self, input_tensors):
        return [in_tensors[5]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor,
                              golden_out_tensor,
                              rtol=0.01,
                              atol=0.01)

    def test(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3],
            in_tensors[4]
        ])


if __name__ == '__main__':
    unittest.main()
