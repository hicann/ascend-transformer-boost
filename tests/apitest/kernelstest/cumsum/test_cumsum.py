#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import unittest
import numpy as np
import torch
import op_test


OP_NAME = "CumsumOperation"


def hillis_steele_fp16(x, axis):
    # CumsumF16Kernel 的 fp16 累加实现是 Hillis-Steele 并行前缀和，
    # golden 必须逐位模拟该算法（实测与算子输出 maxAbs=0），
    # 用顺序 np.cumsum 或 fp32 累加都会与算子输出产生 fp16 ULP 级偏差。
    x = np.array(x, dtype=np.float16, copy=True)
    n = x.shape[axis]
    d = 1
    while d < n:
        cur = [slice(None)] * x.ndim
        cur[axis] = slice(d, None)
        prev = [slice(None)] * x.ndim
        prev[axis] = slice(None, -d)
        x[tuple(cur)] = np.float16(x[tuple(cur)] + x[tuple(prev)])
        d *= 2
    return x


class TestCumsum(op_test.OpTest):
    def golden_calc(self, in_tensors):
        axis = self.op_desc["specificParam"]["axis"][0]
        if in_tensors[0].dtype == torch.bfloat16:
            x = in_tensors[0]
            x = x.to(torch.float32)
            x = x.numpy()
            return [torch.from_numpy(np.cumsum(x, axis=axis))]
        # fp16：模拟算子的 Hillis-Steele fp16 累加路径
        x = in_tensors[0].numpy()
        return [torch.from_numpy(hillis_steele_fp16(x, axis))]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if out_tensors[0].dtype == torch.bfloat16:
            return torch.allclose(out_tensors[0].bfloat16(), golden_out_tensors[0].bfloat16(), rtol=2**-7, atol=2**-7)
        else:
            return torch.allclose(out_tensors[0], golden_out_tensors[0].half(), rtol=2**-8, atol=2**-8)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_cumsum_dtm(self):
        shape = (128, 1111)
        input0 = np.random.uniform(low=-10, high=10, size=shape).astype(np.float16)

        op_param = {"axis": [1], "deterministic": True}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0)], [torch.zeros(shape).half()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_cumsum(self):
        shape = (128, 2222)
        input0 = np.random.uniform(low=-10, high=10, size=shape).astype(np.float16)

        op_param = {"axis": [1]}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0)], [torch.zeros(shape).half()])

    @op_test.only_910b
    def test_cumsum_bf16_dtm(self):
        shape = (128, 1111)
        input0 = np.random.uniform(low=-10, high=10, size=shape).astype(np.float32)

        op_param = {"axis": [1], "deterministic": True}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16()], [torch.zeros(shape).bfloat16()])

    @op_test.only_910b
    def test_cumsum_bf16(self):
        shape = (128, 2222)
        input0 = np.random.uniform(low=-10, high=10, size=shape).astype(np.float32)

        op_param = {"axis": [1]}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16()], [torch.zeros(shape).bfloat16()])


if __name__ == '__main__':
    unittest.main()
