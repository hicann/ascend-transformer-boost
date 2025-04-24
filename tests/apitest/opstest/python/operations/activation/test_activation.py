#
# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import itertools
import os
import sys
import unittest

import numpy as np
import torch
import torch_npu
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "ActivationOperation"
PARAM = {"activationType": 4, "scale": 1.0}
PARAM_FASTER_GELU_FORWARD = {"activationType": 9}
HIGH_NUM = 5
LOW_NUM = -5


class TestActivationOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].float()
        outtensor = float_in_tensors / (1 + torch.exp(-float_in_tensors * PARAM["scale"]))
        return [outtensor.bfloat16()]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        intensor0 = torch.rand(2, 3, 5).npu().bfloat16()
        self.execute(OP_NAME, PARAM, [intensor0])


class TestSwigluForwardOperation(operation_test.OperationTest):
    SPLIT_DIM = -1

    def golden_calc(self, in_tensors):
        x = in_tensors[0]
        a, b = x.chunk(2, dim=self.SPLIT_DIM)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        y = torch.sigmoid(a) * a * b
        y = y.to(torch.float16)
        return [y]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input0 = np.random.uniform(-2, 2, (8192, 1, 3904)).astype("float16")
        self.execute(OP_NAME, {"activationType": 6, "dim": self.SPLIT_DIM}, [torch.from_numpy(input0).npu().half()])

class TestSwigluBackwardOperation(operation_test.OperationTest):
    SPLIT_DIM = -1

    def swiglu(self, x):
        return x * torch.sigmoid(x)

    def swiglu_grad(self, x):
        return torch.sigmoid(x) + x * (1 - torch.sigmoid(x)) * torch.sigmoid(x)

    def golden_calc(self, in_tensors):
        tensor_y_grad = in_tensors[0]
        x = in_tensors[1]
        a, b = x.chunk(2, dim=self.SPLIT_DIM)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        y1 = b * tensor_y_grad * self.swiglu_grad(a)
        y2 = tensor_y_grad * self.swiglu(a)
        y = torch.cat((y1, y2), dim=self.SPLIT_DIM)
        return [y]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input0 = np.random.uniform(-2, 2, (8192, 1, 1952)).astype("float32")
        input1 = np.random.uniform(-2, 2, (8192, 1, 3904)).astype("float32")
        self.execute(OP_NAME, {"activationType": 7, "dim": self.SPLIT_DIM},
                     [torch.from_numpy(input0).npu().float(), torch.from_numpy(input1).npu().float()])

class TestFasterGeluForwardNz310pFp16Operation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].float()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.half()]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64], [1123, 4032]]:
            intensor0 = torch.rand(shape[0], shape[1], dtype=torch.float16).npu().half()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = tensor_nd_to_nz_2d(intensor0)
            intensor0 = torch_npu.npu_format_cast(intensor0, 29)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNz310pFp32Operation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].float()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.float()]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64], [1123, 4032]]:
            intensor0 = torch.rand(shape[0], shape[1]).npu().float()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = tensor_nd_to_nz_2d(intensor0)
            intensor0 = torch_npu.npu_format_cast(intensor0, 29)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNz910Bf16Operation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].bfloat16()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.bfloat16()]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64], [1123, 4032]]:
            intensor0 = torch.rand(shape[0], shape[1]).npu().bfloat16()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = tensor_nd_to_nz_2d(intensor0)
            intensor0 = torch_npu.npu_format_cast(intensor0, 29)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNd310pFp16Operation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].float()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.half()]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64], [1123, 4032]]:
            intensor0 = torch.rand(shape[0], shape[1], dtype=torch.float16).npu().half()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = torch_npu.npu_format_cast(intensor0, 2)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNd310pFp32Operation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].float()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.float()]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64], [1123, 4032]]:
            intensor0 = torch.rand(shape[0], shape[1]).npu().float()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = torch_npu.npu_format_cast(intensor0, 2)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNd910Bf16Operation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].bfloat16()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.bfloat16()]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64], [1123, 4032]]:
            intensor0 = torch.rand(shape[0], shape[1]).npu().bfloat16()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = torch_npu.npu_format_cast(intensor0, 2)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNd910Bf16UnalignOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].bfloat16()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.bfloat16()]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        shapes, base_shape = [], [16, 1024]
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] - 15 * i])
        for shape in shapes:
            intensor0 = torch.rand(shape[0], shape[1]).npu().bfloat16()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = torch_npu.npu_format_cast(intensor0, 2)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNdFp32GeneralizeOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].float()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.float()]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        test_list = [1, 4, 15, 16, 17, 32, 64, 65, 128, 256, 257, 13107]
        generalize_shapes = list(itertools.product(test_list, repeat=2))
        generalize_shapes = [list(pair) for pair in generalize_shapes]
        for shape in generalize_shapes:
            intensor0 = torch.rand(shape[0], shape[1]).npu().float()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = torch_npu.npu_format_cast(intensor0, 2)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNdFp16UnalignOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].float()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.half()]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        shapes, base_shape = [], [16, 1024]
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] - 15 * i])
        for shape in shapes:
            intensor0 = torch.rand(shape[0], shape[1], dtype=torch.float16).npu().half()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = torch_npu.npu_format_cast(intensor0, 2)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


class TestFasterGeluForwardNdFp32UnalignOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        float_in_tensors = in_tensors[0].float()
        float_result = get_golden_data(float_in_tensors)
        return [float_result.float()]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        shapes, base_shape = [], [16, 1024]
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] - 15 * i])
        for shape in shapes:
            intensor0 = torch.rand(shape[0], shape[1]).npu().float()
            intensor0 = intensor0 * (HIGH_NUM - (LOW_NUM)) + LOW_NUM
            intensor0 = torch_npu.npu_format_cast(intensor0, 2)
            self.execute(OP_NAME, PARAM_FASTER_GELU_FORWARD, [intensor0])


def get_golden_data(in_tensors):
    return in_tensors * torch.exp(0.851 * (in_tensors - torch.abs(in_tensors))) / (
            1 + torch.exp(-1.702 * torch.abs(in_tensors)))


def tensor_nd_to_nz_2d(in_tensors):
    ALIGN_INT8 = 32
    DEFAULT_ALIGN = 16

    aux_dims = [0, 0, 0, 0]
    aux_dims[0] = 1
    aux_dims[1] = round_up(in_tensors.size(0), DEFAULT_ALIGN)

    pad_dims = [0, 0, 0, 0]
    pad_dims[3] = round_up(in_tensors.size(0), DEFAULT_ALIGN) - in_tensors.size(0)

    if in_tensors.dtype == torch.int8:
        aux_dims[2] = round_up(in_tensors.size(1), ALIGN_INT8) // ALIGN_INT8
        aux_dims[3] = ALIGN_INT8
        pad_dims[1] = round_up(in_tensors.size(1), ALIGN_INT8) - in_tensors.size(1)
    else:
        aux_dims[2] = round_up(in_tensors.size(1), DEFAULT_ALIGN) // DEFAULT_ALIGN
        aux_dims[3] = DEFAULT_ALIGN
        pad_dims[1] = round_up(in_tensors.size(1), DEFAULT_ALIGN) - in_tensors.size(1)

    return custom_transpose(custom_reshape(custom_pad(in_tensors, pad_dims), aux_dims), 1, 2).contiguous()


def tensor_nd_to_nz_3d(in_tensors):
    ALIGN_INT8 = 32
    DEFAULT_ALIGN = 16

    aux_dims = [0, 0, 0, 0]
    aux_dims[0] = in_tensors.size(0)
    aux_dims[1] = round_up(in_tensors.size(1), DEFAULT_ALIGN)

    pad_dims = [0, 0, 0, 0]
    pad_dims[3] = round_up(in_tensors.size(1), DEFAULT_ALIGN) - in_tensors.size(1)

    if in_tensors[0].dtype == torch.int8:
        aux_dims[2] = round_up(in_tensors.size(2), ALIGN_INT8) // ALIGN_INT8
        aux_dims[3] = ALIGN_INT8
        pad_dims[1] = round_up(in_tensors.size(2), ALIGN_INT8) - in_tensors.size(2)
    else:
        aux_dims[2] = round_up(in_tensors.size(2), DEFAULT_ALIGN) // DEFAULT_ALIGN
        aux_dims[3] = DEFAULT_ALIGN
        pad_dims[1] = round_up(in_tensors.size(2), DEFAULT_ALIGN) - in_tensors.size(2)

    pad_op = custom_pad(in_tensors, pad_dims)
    reshape_op = custom_reshape(pad_op, aux_dims)
    transpose_op = custom_transpose(reshape_op, 1, 2)

    return transpose_op.contiguous()


def tensor_nz_to_nd(in_tensors, outCrops):
    aux_dims = [0, 0, 0]
    aux_dims[0] = in_tensors.size(0)
    aux_dims[1] = in_tensors.size(2)
    aux_dims[2] = in_tensors.size(1) * in_tensors.size(3)
    transpose_op = custom_transpose(in_tensors, 1, 2)
    reshape_op = custom_reshape(transpose_op, aux_dims)
    split_op = reshape_op[:, :outCrops[0], :outCrops[1]]
    return split_op


def round_up(x, align):
    if align == 0:
        return -1
    return (x + align - 1) // align * align


def custom_transpose(x, dim1, dim2):
    return x.transpose(dim1, dim2)


def custom_pad(x, pad_dims):
    return torch.nn.functional.pad(x, pad_dims)


def custom_reshape(x, target_shape):
    return x.reshape(target_shape)


if __name__ == '__main__':
    unittest.main()
