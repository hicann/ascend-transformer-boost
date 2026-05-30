#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
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
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "SwigluQuantOperation"
PARAM = {"quantType": 0}


class TestSwigluQuantOperation(operation_test.OperationTest):
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _swiglu(a, b):
        return TestSwigluQuantOperation._sigmoid(a) * a * b

    @staticmethod
    def _prepare_input(x):
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x_np = np.array(x.cpu().float()).astype(np.float32)
        a, b = np.split(x_np, 2, axis=1)
        return TestSwigluQuantOperation._swiglu(a, b)

    @staticmethod
    def _golden_910b(swiglu_y):
        """910B OpsRunner: round int8, dequant scale (y_max/127)."""
        y_tmp = np.array(swiglu_y)
        y_max = np.amax(np.abs(y_tmp), axis=1)
        scale_tmp = 127.0 / y_max
        scale = scale_tmp.reshape(-1, 1)
        quant_y = np.round(y_tmp * scale).astype(np.int8)
        scale_out = (1.0 / scale).reshape(-1)
        return quant_y, scale_out.astype(np.float32)

    @staticmethod
    def _golden_950(swiglu_y):
        """950 AclnnRunner: clip int8 (ACLNN kernel); dequant scale after InplaceReciprocal."""
        y_tmp = np.array(swiglu_y).astype(np.float32)
        y_max = np.amax(np.abs(y_tmp), axis=1)
        scale_tmp = 127.0 / y_max
        scale = scale_tmp.reshape(-1, 1)
        quant_y = np.clip(np.round(y_tmp * scale), -128, 127).astype(np.int8)
        scale_out = (1.0 / scale).reshape(-1)
        return quant_y, scale_out.astype(np.float32)

    def golden_calc(self, in_tensors):
        swiglu_y = self._prepare_input(in_tensors[0])
        soc = operation_test.get_soc_version()
        if soc == "Ascend950":
            quant_y, scale_out = self._golden_950(swiglu_y)
        else:
            quant_y, scale_out = self._golden_910b(swiglu_y)
        return [
            torch.from_numpy(quant_y).to("cpu"),
            torch.from_numpy(scale_out).to("cpu"),
        ]

    def golden_compare(self, out_tensor, golden_out_tensor):
        actual_output = out_tensor.cpu()
        golden_output = golden_out_tensor.cpu()
        if actual_output.dtype == torch.int8:
            diff = np.abs(actual_output.numpy() - golden_output.numpy())
            return not (diff > 1).any()
        if actual_output.dtype == torch.float32:
            return np.allclose(actual_output.numpy(), golden_output.numpy(), rtol=0.0001, atol=0.0001)
        return False

    def _run_case(self):
        input_token_num = 128
        input_hidden_size = 4096
        shape = (input_token_num, input_hidden_size)
        x = torch.empty(shape).uniform_(0, 1).to(torch.float16)
        self.execute(OP_NAME, PARAM, [x.npu().half()])

    def test_swi_glu_quant_910b(self):
        if operation_test.get_soc_version() != "Ascend910B":
            print("this testcase only supports Ascend910B")
            return
        self._run_case()

    def test_swi_glu_quant_950(self):
        if operation_test.get_soc_version() != "Ascend950":
            print("this testcase only supports Ascend950")
            return
        self._run_case()


if __name__ == "__main__":
    unittest.main()
