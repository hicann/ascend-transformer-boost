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
import op_test
import logging

BLOCK_SIZE = 16

class TestMatmulF32(op_test.OpTest):
    def __gen_test_data(self, shpae: tuple) -> None:
        bsize, msize, ksize, nsize = shpae
        bat_A, bat_B, bat_C = [], [], []
        op_param = self.op_desc["specificParam"]
        input_formats = self.op_desc["input_formats"]
        for _ in range(bsize):
            a = np.random.uniform(-2, 2, size=(msize, ksize)).astype(np.float32)
            b = np.random.uniform(-2, 2, size=(ksize, nsize)).astype(np.float32)
            c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.float32)
            bat_A.append(a)
            bat_B.append(b)
            bat_C.append(c)
        self.bat_A = np.stack(bat_A)
        self.bat_B = np.stack(bat_B)
        self.bat_C = np.stack(bat_C)
        return

    def golden_calc(self, in_tensors):
        return [torch.tensor(self.bat_C).float()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

    @op_test.only_910b
    def testcase0(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 1, 8192, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase1(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 1, 456, 789
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase2(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 1, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase3(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 5, 1, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase4(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 4, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase5(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 5, 16, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase6(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 5, 16, 345, 657
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase7(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 2, 2, 80, 8192
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase8(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 129, 1024, 8192
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    # @op_test.only_910b
    # def testcase9(self):
    #     self.trans_A, self.trans_B = False, False
    #     bsize, msize, ksize, nsize = 1, 204800, 80, 8192
    #     self.set_param(
    #         "MatMulOperation",
    #         {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
    #     )
    #     self.set_input_formats([self.format_nd, self.format_nd])
    #     self.set_output_formats([self.format_nd])
    #     self.__gen_test_data((bsize, msize, ksize, nsize))
    #     self.execute(
    #         [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
    #         [torch.zeros(self.bat_C.shape).float()],
    #         {"ASDOPS_MATMUL_PP_FLAG": "0"},
    #     )

    # @op_test.only_910b
    # def testcase11(self):
    #     self.trans_A, self.trans_B = False, False
    #     bsize, msize, ksize, nsize = 1, 143962, 64, 64
    #     self.set_param(
    #         "MatMulOperation",
    #         {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
    #     )
    #     self.set_input_formats([self.format_nd, self.format_nd])
    #     self.set_output_formats([self.format_nd])
    #     self.__gen_test_data((bsize, msize, ksize, nsize))
    #     self.execute(
    #         [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
    #         [torch.zeros(self.bat_C.shape).float()],
    #         {"ASDOPS_MATMUL_PP_FLAG": "0"},
    #     )

    @op_test.only_910b
    def testcase_llama65b_1(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 167, 8192, 3072
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase_llama65b_2(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 167, 8192, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase_llama65b_3(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 167, 8192, 5504
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase_llama65b_4(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 167, 8192, 2752
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase_llama65b_5(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 167, 2752, 8192
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

    @op_test.only_910b
    def testcase_llama65b_6(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 167, 8192, 32000
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).float(), torch.tensor(self.bat_B).float()],
            [torch.zeros(self.bat_C.shape).float()],
            {"ASDOPS_MATMUL_PP_FLAG": "0"},
        )

if __name__ == "__main__":
    unittest.main()
