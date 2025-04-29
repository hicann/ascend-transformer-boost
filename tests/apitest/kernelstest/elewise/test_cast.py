#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import unittest
import numpy as np
import torch
import sys
import logging
import op_test

OP_NAME = "ElewiseOperation"
OP_PARAM_CAST = {"elewiseType": 1,"outTensorType": 3}

class TestElewise(op_test.OpTest):
    def golden_calc(self, in_tensors):
        if self.op_desc["specificParam"]["outTensorType"] == 3:             # int32
            in_tensors = in_tensors[0].type(torch.int32)
 
        elif self.op_desc["specificParam"]["outTensorType"] == 1:           # fp16
            in_tensors = in_tensors[0].type(torch.half)

        elif self.op_desc["specificParam"]["outTensorType"] == 0:           # fp32
            in_tensors = in_tensors[0].type(torch.float32)

        elif self.op_desc["specificParam"]["outTensorType"] == 27:          # bf16
            in_tensors = in_tensors[0].type(torch.bfloat16)

        elif self.op_desc["specificParam"]["outTensorType"] == 9:          # int64
            in_tensors = in_tensors[0].type(torch.int64)

        return [in_tensors]

    def golden_compare(self, out_tensors, golden_out_tensors):
        self.assertTrue(len(out_tensors) == len(golden_out_tensors))
        result = []
        if self.op_desc["specificParam"]["outTensorType"] == 1 or \
            self.op_desc["specificParam"]["outTensorType"] == 0 or \
            self.op_desc["specificParam"]["outTensorType"] == 27:
            logging.debug("float16/bfloat16/float32 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                logging.debug(f"actual_output is : {actual_output.type(torch.float32)}")
                logging.debug(f"golden_output is : {golden_output.type(torch.float32)}")
                result.append(torch.equal(actual_output.type(torch.float32), golden_output.type(torch.float32)))
        elif self.op_desc["specificParam"]["outTensorType"] == 3:
            logging.debug("int32 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                result.append(torch.equal(actual_output.type(torch.int32), golden_output.type(torch.int32)))
        elif self.op_desc["specificParam"]["outTensorType"] == 9:
            logging.debug("int64 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                result.append(torch.equal(actual_output.type(torch.int64), golden_output.type(torch.int64)))
        else:
            logging.debug("Unsupport dtype:%s golden compare", actual_output.dtype)
            result.append(False)

        logging.debug(f"result is : {all(result)}")
        return all(result)

    @op_test.only_910b
    def test_cast_fp32_to_bf16(self):
        shape = (10000)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.float32)
        OP_PARAM_CAST["outTensorType"] = 27
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.bfloat16)])

    @op_test.only_910b
    def test_cast_bf16_to_fp32(self):
        shape = (10000)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.float32)
        OP_PARAM_CAST["outTensorType"] = 0
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0).type(torch.bfloat16)],
                    [torch.zeros(shape).type(torch.float32)])

    @op_test.skip_310b
    def test_cast_fp32_to_i32(self):
        shape = (10000)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.float32)
        OP_PARAM_CAST["outTensorType"] = 3
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.int32)])

    @op_test.skip_310b
    def test_cast_int32_to_half(self):
        shape = (10000)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.int32)
        OP_PARAM_CAST["outTensorType"] = 1
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.half)])

    @op_test.skip_310b
    def test_cast_half_to_i32(self):
        shape = (10000)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.half)
        OP_PARAM_CAST["outTensorType"] = 3
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.int32)])

    def test_cast_fp16_to_fp32(self):
        shape = (10000)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.float16)
        OP_PARAM_CAST["outTensorType"] = 0
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.float32)])

    def test_cast_fp32_to_fp16(self):
        shape = (10000)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.float32)
        OP_PARAM_CAST["outTensorType"] = 1
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.float16)])

    @op_test.only_910b
    def test_cast_i64_to_i32(self):
        shape = (5440)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.int64)
        OP_PARAM_CAST["outTensorType"] = 3
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.int32)])

    @op_test.only_910b
    def test_cast_i32_to_f16(self):
        shape = (5440)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.int32)
        OP_PARAM_CAST["outTensorType"] = 1
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.float16)])

    @op_test.only_910b
    def test_cast_i32_to_i64(self):
        torch.set_printoptions(profile="full")
        shape = (5440)
        input0 = np.random.uniform(low=1, high=100, size=shape).astype(np.int32)
        OP_PARAM_CAST["outTensorType"] = 9
        self.set_param(OP_NAME, OP_PARAM_CAST)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).type(torch.int64)])

if __name__ == '__main__':
    unittest.main()