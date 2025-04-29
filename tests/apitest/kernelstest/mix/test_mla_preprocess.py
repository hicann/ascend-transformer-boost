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
import torch_npu
import torch.nn.functional as F
import sys, os
import op_test
import random
import logging

sys.path.append("../")
sys.path.append("../..")
from precision_calcu import *

OP_NAME = "MlaPreprocessOperation"
QUANTMAX = 127
QUANTMIN = -128
block_size = 128

random.seed(12)
np.random.seed(12)
torch.manual_seed(12)


def process_deq_scale(deq_scale: torch.Tensor) -> np.ndarray:
    ret = torch.frombuffer(deq_scale.numpy().tobytes(), dtype=torch.int32).to(torch.int64)
    return ret


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, ((0, r_pad, 0, c_pad)))
    nz_mat = torch.permute(
        torch.reshape(nd_mat, (r // block_size[0], block_size[0], c // block_size[1], block_size[1])), [2, 0, 1, 3]
    )
    nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def transdata_3d(nd_mat, block_size: tuple = (16, 16)):
    if nd_mat.ndim != 3:
        raise ValueError("Expected a 3-dimensional input array.")
    B, K, N = nd_mat.shape
    processed_slices = []

    for batch_index in range(B):
        current_slice = nd_mat[batch_index]
        nz_mat = transdata(current_slice, block_size)
        processed_slices.append(nz_mat)

    result = torch.stack(processed_slices, axis=0)
    return result


class TestMLAPrepross(op_test.OpTest):
    def __set_envs(self, env: dict):
        if env:
            for key, value in env.items():
                os.environ[key] = value

    def __unset_envs(self, env: dict):
        if env:
            for key, _ in env.items():
                os.environ[key] = ""

    def __get_npu_device(self):
        npu_device = os.environ.get("MKI_NPU_DEVICE")
        if npu_device is None:
            npu_device = "npu:0"
        else:
            npu_device = f"npu:{npu_device}"
        return npu_device

    def padHeadNum(self, headdim_sin_cos):
        return torch.tile(headdim_sin_cos, (1, self.headNum))

    def rotateHalfX(self, q_temp):
        # 拆分成 head_num 个 [n,head_dim] 的二维向量
        # print("============== q_temp", q_temp)
        # import pdb;pdb.set_trace()
        q_splits = torch.chunk(q_temp, self.headNum, dim=1)
        # 对每个 [n,head_dim] 向量的第二维进行分割，并对第二块乘以 -1再拼回到第一块前面
        # import pdb;pdb.set_trace()
        processed_q_splits = []
        for q_split in q_splits:
            # print("============== q_split", q_split)
            # 分割第二维
            first_half, second_half = torch.chunk(q_split, 2, dim=1)
            # 拼接回 [n,head_dim] 的二维向量
            # import pdb;pdb.set_trace()
            processed_q_split = torch.cat((-second_half, first_half), dim=1)
            processed_q_splits.append(processed_q_split)

        # 将所有处理后的 [n,head_dim] 向量拼回 [n,head_num*head_dim] 的二维向量
        return torch.cat(processed_q_splits, dim=1)

    def rotateHalf2(self, q_temp):
        # 拆分成 head_num 个 [n,head_dim] 的二维向量
        # print("============== q_temp", q_temp)
        q_splits = np.split(q_temp, self.headNum, axis=1)
        # 对每个 [n,head_dim] 向量的第二维进行分割，并对第二块乘以 -1再拼回到第一块前面
        processed_q_splits = []
        for q_split in q_splits:
            # print("============== q_split", q_split)
            # 分割第二维
            first_half, second_half = np.split(q_split, 2, axis=1)
            # 拼接回 [n,head_dim] 的二维向量
            processed_q_split = np.concatenate((second_half, first_half), axis=1)
            processed_q_splits.append(processed_q_split)

        # 将所有处理后的 [n,head_dim] 向量拼回 [n,head_num*head_dim] 的二维向量
        return np.concatenate(processed_q_splits, axis=1)

    def RopeConcatGolden(self, q, sin, cos, concatInput):
        pad_sin = self.padHeadNum(sin)
        pad_cos = self.padHeadNum(cos)
        # import pdb;pdb.set_trace()
        # self.qcosOut = self.rotateHalf(q)
        rope_res = q * pad_cos + self.rotateHalfX(q) * pad_sin
        rope_res = rope_res.reshape(self.input_token_num, self.headNum, self.rope_hidden_size)
        rope_res = rope_res.to(self.dtype)

        return torch.cat((concatInput.to(self.dtype), rope_res), dim=2)

    def rmsNormGolden(self, x, gamma):
        x_float32 = x.to(torch.float32)
        square_sum = torch.sum(torch.square(x_float32), axis=-1, keepdims=True)
        rms = 1.0 / torch.sqrt(square_sum / self.rms_hidden_size + self.epsilon)
        gamma_float32 = gamma.to(torch.float32)
        rmsNorm = rms * x_float32 * gamma_float32
        result = rmsNorm.to(self.dtype)
        np.set_printoptions(suppress=True, formatter={"float_kind": "{:.15f}".format})
        return result

    def rotateHalf(self, k_temp):
        first_half, second_half = torch.chunk(k_temp, 2, dim=1)
        processed_k_split = torch.cat((-second_half, first_half), dim=1)
        return processed_k_split

    def RopeGolden(self, keyRope, sin, cos):
        RopeGolden = keyRope * cos + self.rotateHalf(keyRope) * sin
        return RopeGolden

    def RACGolden(self, keyRAC, slotMapping, keycacheout_golden):
        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size
            token_key = keyRAC[i]

            keycacheout_golden[block_index][block_offset] = token_key
        return keycacheout_golden

    def RmsNormAndRopeAndReshapeAndCacheGolden(self, x, gamma, keyRope, cos, sin, slotMapping, keycachein):
        rmsNormOutput = self.rmsNormGolden(x, gamma)
        ropeOutput = self.RopeGolden(keyRope, sin, cos)
        ropeReshape = ropeOutput.reshape(self.input_token_num, 1, self.rope_hidden_size)

        keyRAC = torch.cat((rmsNormOutput, ropeReshape), axis=-1)
        return self.RACGolden(keyRAC, slotMapping, keycachein)

    def rms_norm_quant_calc(self, input, gamma, beta, quantScale, quantOffset):
        out_shape = input.shape
        scale = 1.0 / quantScale.item()
        offset = quantOffset.item()
        inputScale = torch.tensor(scale, dtype=torch.float)
        inputOffset = torch.tensor(offset, dtype=torch.float)
        input0 = torch.tensor(input).float()
        input1 = torch.tensor(gamma).float()
        square_sum = torch.sum(torch.square(input0), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / out_shape[-1] + self.epsilon)
        output = input0 * factor * input1
        output = (output + beta.float()) * inputScale + inputOffset
        self.xxTest = output.clone()
        output = torch.round(output)
        output = torch.tensor(output).to(torch.float16)
        output = torch.min(output, torch.tensor(QUANTMAX, dtype=torch.half))
        output = torch.max(output, torch.tensor(QUANTMIN, dtype=torch.half))
        return torch.tensor(output).to(torch.int8)

    def ein_sum_out_quant_golden(self, input, scale):
        if (scale.dtype == torch.bfloat16):
            input = input.float()
            scale = scale.float()
        quant = input * scale
        output = torch.round(quant.float()).half()
        output = torch.min(output, torch.tensor(QUANTMAX, dtype=torch.half))
        output = torch.max(output, torch.tensor(QUANTMIN, dtype=torch.half))
        return output.to(torch.int8)



    def calc_vec_mm_atb_data(self, N, headNum, data_type, cacheMode):
        hiddenStrate = 7168
        blockNum = 192
        blockSize = 128
        headdim = 576
        self.input_token_num = N
        self.rms_hidden_size = 512
        self.rope_hidden_size = 64
        self.headNum = headNum
        self.epsilon = 1e-6
        self.dtype = data_type

        self.input1 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(N, 7168))).to(data_type)  #
        self.gamma1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(hiddenStrate))).to(data_type)
        self.quantScale1 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.quantOffset1 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1))).to(torch.int8)
        self.wdqkv = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(2112, 7168))).to(torch.int8)  #

        if cacheMode == 2:
            self.I8keyCache1 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(blockNum, blockSize, 1, 512))).to(torch.int8)
        elif cacheMode ==3:
            self.FPkeyCache1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(blockNum, blockSize, 1, 512))).to(data_type)
        self.keyCache2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(blockNum, blockSize, 1, 64))).to(data_type) #[..., 512:576]
        self.deScale1 = torch.rand((2112), dtype=torch.float32) / 1000
        self.gamma2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(1536))).to(data_type)
        self.quantScale2 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.quantOffset2 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1))).to(torch.int8)

        self.wuq = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(headNum * 192, 1536))).to(torch.int8)  #

        self.deScale2 = torch.rand((headNum * 192), dtype=torch.float32) / 1000

        self.gamma3 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(512))).to(data_type)
        self.sin1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)
        self.cos1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)
        self.keyCache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(blockNum, blockSize, 1, headdim))).to(
            data_type
        )

        self.slotMapping = torch.from_numpy(np.random.choice(192 * 128, N, replace=False).astype(np.int32)).to(
            torch.int32
        )
        self.slotMapping[0] = -1
        self.wuk = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(headNum, 128, 512))).to(data_type)  #
        self.sin2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)
        self.cos2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)

        self.bias1 = torch.from_numpy(np.random.randint(-10, 10, (1, 2112)).astype(np.int32)).to(torch.int32)
        self.bias2 = torch.from_numpy(np.random.randint(-10, 10, (1, headNum * 192)).astype(np.int32)).to(torch.int32)

        self.beta1 = torch.from_numpy(np.random.randint(-2, 2, (hiddenStrate)).astype(np.float16)).to(data_type)
        self.beta2 = torch.from_numpy(np.random.randint(-2, 2, (1536)).astype(np.float16)).to(data_type)

        self.qNopeScale = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(1, headNum, 1))).to(data_type)
        self.quantScale3 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.calc_vec_mm_data(N, headNum, data_type)

        ## RmsNorm
        self.rms1Out1_npu = torch.zeros((N, 7168), dtype=torch.int8)
        npu_device = self.__get_npu_device()
        torch_npu.npu.set_device(npu_device)
        in_tensors = [self.input1, self.gamma1, self.beta1, self.quantScale1, self.quantOffset1]
        out_tensors = [self.rms1Out1_npu]
        OP_NAME = "NormOperation"
        OP_PARAM0 = {"normType": 2, "inGamma": True, "inBeta": True, "epsilon": 1e-6}
        self.set_param(OP_NAME, OP_PARAM0)
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.rms1Out1_npu = out_tensors_npu[0].cpu().clone()

        ##Ppmatmul
        self.mm1Out1_npu = torch.zeros((N, 2112), dtype=data_type)
        in_tensors = [out_tensors_npu[0], self.wdqkv, self.bias1, self.deScale1]
        out_tensors = [self.mm1Out1_npu]
        self.set_param(
            "MatMulOperation", {"transposeA": False, "transposeB": True, "withBias": True, "enDequant": True}
        )
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        self.__set_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.__unset_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mm1Out1_npu = out_tensors_npu[0].cpu().clone()

        ##SplitV
        splitSize = [512, 64, 1536]
        splitVDim = 1
        OP_NAME = "SplitOperation"
        OP_PARAM = {"splitNum": 3, "splitVDim": [splitVDim], "splitSize": splitSize}
        self.set_param(OP_NAME, OP_PARAM)
        in_tensors_npu = [out_tensors_npu[0]]
        Split1_out_tensors_npu = []
        shape = in_tensors_npu[0].shape
        for size in splitSize:
            slice_shape = list(shape)
            slice_shape[splitVDim] = size
            Split1_out_tensors_npu.append(torch.zeros(slice_shape, dtype=data_type).npu())

        self.mki.execute(in_tensors_npu, Split1_out_tensors_npu)
        ## RmsNorm
        self.rms2Out_npu = torch.zeros((N, 1536), dtype=torch.int8)
        in_tensors = [Split1_out_tensors_npu[2], self.gamma2, self.beta2, self.quantScale2, self.quantOffset2]
        out_tensors = [self.rms2Out_npu]
        OP_NAME = "NormOperation"
        OP_PARAM0 = {"normType": 2, "inGamma": True, "inBeta": True, "epsilon": 1e-6}
        self.set_param(OP_NAME, OP_PARAM0)
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.rms2Out_npu = out_tensors_npu[0].cpu().clone()

        # ##Ppmatmul
        self.mm2Out_npu = torch.zeros((N, headNum * 192), dtype=data_type)
        in_tensors = [out_tensors_npu[0], self.wuq, self.bias2, self.deScale2]
        out_tensors = [self.mm2Out_npu]
        self.set_param(
            "MatMulOperation", {"transposeA": False, "transposeB": True, "withBias": True, "enDequant": True}
        )
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        mm2out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        self.__set_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mki.execute(in_tensors_npu, mm2out_tensors_npu)
        self.__unset_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mm2Out_npu = mm2out_tensors_npu[0].cpu().clone()

        # # ##SplitV
        splitSize = [128, 64]
        splitVDim = 2
        OP_NAME = "SplitOperation"
        OP_PARAM = {"splitNum": 2, "splitVDim": [splitVDim], "splitSize": splitSize}
        self.set_param(OP_NAME, OP_PARAM)
        in_tensors = mm2out_tensors_npu[0].reshape(N, headNum, 192)
        in_tensors_npu = in_tensors.npu()
        Split2_out_tensors_npu = []
        shape = in_tensors_npu.shape
        for size in splitSize:
            slice_shape = list(shape)
            slice_shape[splitVDim] = size
            Split2_out_tensors_npu.append(torch.zeros(slice_shape, dtype=data_type).npu())
        self.mki.execute([in_tensors_npu], Split2_out_tensors_npu)

        # EinSum
        # EinSumInput = torch.transpose(Split2_out_tensors_npu[0], 0, 1)
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = headNum, N, 128, 512
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
            },
        )

        in_tensors = [Split2_out_tensors_npu[0], self.wuk]
        Einsumout_tensors = [torch.zeros((N, headNum, 512), dtype=data_type)]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        Einsumout_tensors_npu = [tensor.npu() for tensor in Einsumout_tensors]
        self.__set_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mki.execute(in_tensors_npu, Einsumout_tensors_npu)
        self.__unset_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})

        # Einsumout_tensors_npu = [torch.transpose(Einsumout_tensors_npu[0], 0, 1)]
        self.einsumOut = Einsumout_tensors_npu[0].cpu().clone()

        self.qOut_npu = torch.zeros((N, headNum, 576), dtype=data_type)
        OP_NAME = "RopeQConcatOperation"
        OP_PARAM0 = {}
        self.set_param(OP_NAME, OP_PARAM0)
        Split2_out_tensors_npu[1] = Split2_out_tensors_npu[1].cpu().reshape(N, headNum * 64)
        in_tensors = [Split2_out_tensors_npu[1], self.cos2, self.sin2, Einsumout_tensors_npu[0]]
        qOut_tensors = [self.qOut_npu]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        qout_tensors_npu = [qOut_tensors[i] if isinstance(i, int) else i.npu() for i in qOut_tensors]
        self.mki.execute(in_tensors_npu, qout_tensors_npu)
        qout_tensors_npu[0] = torch.cat(
            [qout_tensors_npu[0].cpu()[:, :, 64:], qout_tensors_npu[0].cpu()[:, :, :64]], dim=2
        )
        self.qOut_npu = qout_tensors_npu[0].cpu().clone()

        # cache mode 2 qOut nope quant
        qOutNopeInput = self.qOut_npu[:,:,0:512]
        scale = self.qNopeScale.cpu().clone()
        self.qOutNopeQuant = self.ein_sum_out_quant_golden(qOutNopeInput.clone(), scale)

        # #Rope
        OP_NAME = "RopeOperation"
        rotaryCoeff = 2
        OP_PARAM0 = {"rotaryCoeff": rotaryCoeff}
        self.RopeOut0 = torch.zeros((N, 64), dtype=data_type)
        self.RopeOut1 = torch.zeros((N, 64), dtype=data_type)
        seqlen = torch.randint(1, 2, (N, 1), dtype=torch.int32)
        in_tensors = [
            Split1_out_tensors_npu[1].reshape(N, 1 * 64),
            Split1_out_tensors_npu[1].reshape(N, 1 * 64),
            self.cos1,
            self.sin1,
            seqlen,
        ]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        Ropeout_tensors_npu = [self.RopeOut0.npu(), self.RopeOut1.npu()]
        self.set_param(OP_NAME, OP_PARAM0)
        self.mki.execute(in_tensors_npu, Ropeout_tensors_npu)

        # RmsNorm
        OP_NAME = "NormOperation"
        OP_PARAM0 = {"normType": 2, "inGamma": True, "epsilon": 1e-6, "precisionMode": 0, "gemmaMode": 0}

        in_tensors = [Split1_out_tensors_npu[0].reshape(N, 1, 512), self.gamma3]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        self.RmsNormOut = torch.zeros((N, 1, 512), dtype=data_type).npu()
        RmsNormOut_tensors_npu = [self.RmsNormOut]
        self.set_param(OP_NAME, OP_PARAM0)
        self.mki.execute(in_tensors_npu, RmsNormOut_tensors_npu)
        # kcache1, kcache2
        RmsNorm3Out = RmsNormOut_tensors_npu[0].cpu().clone()
        keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)
        if cacheMode == 2:
            #quant
            quantOut = self.quant(RmsNorm3Out, self.quantScale3)

            nz = True
            # reshape
            if nz:
                self.keyCache1_out = self.reshapeAndCacheNz(quantOut, self.I8keyCache1, self.slotMapping, 512, 32, 16)
                keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)
                self.keyCache2_out = self.reshapeAndCacheNz(keyRope, self.keyCache2, self.slotMapping,64, 16, 4)
            else:
                self.keyCache1_out = self.reshapeAndCache(quantOut, self.I8keyCache1, self.slotMapping, 512)
                keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)
                self.keyCache2_out = self.reshapeAndCache(keyRope, self.keyCache2, self.slotMapping, 64)
        elif cacheMode == 3:
            print("================cacheMode:3")
            self.keyCache1_out = self.reshapeAndCacheNz(RmsNorm3Out, self.FPkeyCache1, self.slotMapping, 512, 16, 32)
            keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)
            self.keyCache2_out = self.reshapeAndCacheNz(keyRope, self.keyCache2, self.slotMapping,64, 16, 4)
        else:
            out_tensors_npu = RmsNormOut_tensors_npu
            # Concat
            OP_NAME = "ConcatOperation"
            OP_PARAM = {"concatDim": 2}
            in_tensors = [RmsNormOut_tensors_npu[0], Ropeout_tensors_npu[0].cpu().reshape(N, 1, 64)]
            ConCat2out_tensors = [torch.zeros((N, 1, 576), dtype=data_type)]
            in_tensors_npu = [tensor.npu() for tensor in in_tensors]
            ConCat2out_tensors_npu = [tensor.npu() for tensor in ConCat2out_tensors]
            self.set_param(OP_NAME, OP_PARAM)
            self.mki.execute(in_tensors_npu, ConCat2out_tensors_npu)

            # Reshape&Cache
            OP_NAME = "ReshapeAndCacheOperation"
            OP_PARAM = {"type": 4}
            in_tensors = [ConCat2out_tensors_npu[0], self.keyOutTensor, self.slotMapping]
            out_tensors = [1]
            in_tensors_npu = [tensor.npu() for tensor in in_tensors]
            out_tensors_npu = [in_tensors_npu[i] if isinstance(i, int) else i.npu() for i in out_tensors]
            self.set_param(OP_NAME, OP_PARAM)
            self.mki.execute(in_tensors_npu, out_tensors_npu)
            self.keyout_npu = out_tensors_npu[0].cpu().clone()


    def calc_vec_mm_atb_data_bf16(self, N, headNum, data_type,cacheMode):
        hiddenStrate = 7168
        blockNum = 192
        blockSize = 128
        headdim = 576
        self.input_token_num = N
        self.rms_hidden_size = 512
        self.rope_hidden_size = 64
        self.headNum = headNum
        self.epsilon = 1e-6
        self.dtype = data_type

        self.input1 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(N, 7168))).to(data_type)  #
        self.gamma1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(hiddenStrate))).to(data_type)
        self.quantScale1 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.quantOffset1 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1))).to(torch.int8)
        self.wdqkv = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(2112, 7168))).to(torch.int8)  #

        self.deScale1 = torch.rand((2112), dtype=torch.float32) / 1000
        self.gamma2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(1536))).to(data_type)
        self.quantScale2 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.quantOffset2 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1))).to(torch.int8)

        self.wuq = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(headNum * 192, 1536))).to(torch.int8)  #

        self.deScale2 = torch.rand((headNum * 192), dtype=torch.float32) / 1000

        self.gamma3 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(512))).to(data_type)
        self.sin1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)
        self.cos1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)
        self.keyCache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(blockNum, blockSize, 1, headdim))).to(
            data_type
        )

        self.slotMapping = torch.from_numpy(np.random.choice(192 * 128, N, replace=False).astype(np.int32)).to(
            torch.int32
        )
        self.slotMapping[0] = -1
        if cacheMode == 2:
            self.I8keyCache1 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(blockNum, blockSize, 1, 512))).to(torch.int8)
        elif cacheMode ==3:
            self.FPkeyCache1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(blockNum, blockSize, 1, 512))).to(data_type)
        self.keyCache2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(blockNum, blockSize, 1, 64))).to(data_type) 


        self.wuk = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(headNum, 128, 512))).to(data_type)  #
        self.sin2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)
        self.cos2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)

        self.bias1 = torch.from_numpy(np.random.randint(-10, 10, (1, 2112)).astype(np.int32)).to(torch.int32)
        self.bias2 = torch.from_numpy(np.random.randint(-10, 10, (1, headNum * 192)).astype(np.int32)).to(torch.int32)

        self.beta1 = torch.from_numpy(np.random.randint(-2, 2, (hiddenStrate)).astype(np.float16)).to(data_type)
        self.beta2 = torch.from_numpy(np.random.randint(-2, 2, (1536)).astype(np.float16)).to(data_type)

        self.qNopeScale = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(1, headNum, 1))).to(data_type)
        self.quantScale3 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)

        self.calc_vec_mm_data(N, headNum, data_type)

        ## RmsNorm
        self.rms1Out1_npu = torch.zeros((N, 7168), dtype=torch.int8)
        npu_device = self.__get_npu_device()
        torch_npu.npu.set_device(npu_device)
        in_tensors = [self.input1, self.gamma1, self.beta1, self.quantScale1, self.quantOffset1]
        out_tensors = [self.rms1Out1_npu]
        OP_NAME = "NormOperation"
        OP_PARAM0 = {"normType": 2, "inGamma": True, "inBeta": True, "epsilon": 1e-6}
        self.set_param(OP_NAME, OP_PARAM0)
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.rms1Out1_npu = out_tensors_npu[0].cpu().clone()

        ##Ppmatmul
        self.mm1Out1_npu = torch.zeros((N, 2112), dtype=data_type)
        in_tensors = [out_tensors_npu[0], self.wdqkv, self.bias1, self.deScale1]
        out_tensors = [self.mm1Out1_npu]
        self.set_param(
            "MatMulOperation",
            {"transposeA": False, "transposeB": True, "withBias": True, "enDequant": True, "outDtype": 27},
        )
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        self.__set_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.__unset_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mm1Out1_npu = out_tensors_npu[0].cpu().clone()

        ##SplitV
        splitSize = [512, 64, 1536]
        splitVDim = 1
        OP_NAME = "SplitOperation"
        OP_PARAM = {"splitNum": 3, "splitVDim": [splitVDim], "splitSize": splitSize}
        self.set_param(OP_NAME, OP_PARAM)
        in_tensors_npu = [out_tensors_npu[0]]
        Split1_out_tensors_npu = []
        shape = in_tensors_npu[0].shape
        for size in splitSize:
            slice_shape = list(shape)
            slice_shape[splitVDim] = size
            Split1_out_tensors_npu.append(torch.zeros(slice_shape, dtype=data_type).npu())

        self.mki.execute(in_tensors_npu, Split1_out_tensors_npu)
        ## RmsNorm
        self.rms2Out_npu = torch.zeros((N, 1536), dtype=torch.int8)
        in_tensors = [Split1_out_tensors_npu[2], self.gamma2, self.beta2, self.quantScale2, self.quantOffset2]
        out_tensors = [self.rms2Out_npu]
        OP_NAME = "NormOperation"
        OP_PARAM0 = {"normType": 2, "inGamma": True, "inBeta": True, "epsilon": 1e-6}
        self.set_param(OP_NAME, OP_PARAM0)
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.rms2Out_npu = out_tensors_npu[0].cpu().clone()

        # ##Ppmatmul
        self.mm2Out_npu = torch.zeros((N, headNum * 192), dtype=data_type)
        in_tensors = [out_tensors_npu[0], self.wuq, self.bias2, self.deScale2]
        out_tensors = [self.mm2Out_npu]
        self.set_param(
            "MatMulOperation",
            {"transposeA": False, "transposeB": True, "withBias": True, "enDequant": True, "outDtype": 27},
        )
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        mm2out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        self.__set_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mki.execute(in_tensors_npu, mm2out_tensors_npu)
        self.__unset_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mm2Out_npu = mm2out_tensors_npu[0].cpu().clone()

        # # ##SplitV
        splitSize = [128, 64]
        splitVDim = 2
        OP_NAME = "SplitOperation"
        OP_PARAM = {"splitNum": 2, "splitVDim": [splitVDim], "splitSize": splitSize}
        self.set_param(OP_NAME, OP_PARAM)
        in_tensors = mm2out_tensors_npu[0].reshape(N, headNum, 192)
        in_tensors_npu = in_tensors.npu()
        Split2_out_tensors_npu = []
        shape = in_tensors_npu.shape
        for size in splitSize:
            slice_shape = list(shape)
            slice_shape[splitVDim] = size
            Split2_out_tensors_npu.append(torch.zeros(slice_shape, dtype=data_type).npu())
        self.mki.execute([in_tensors_npu], Split2_out_tensors_npu)

        # EinSum
        # EinSumInput = torch.transpose(Split2_out_tensors_npu[0], 0, 1)
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = headNum, N, 128, 512
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
            },
        )

        in_tensors = [Split2_out_tensors_npu[0], self.wuk]
        Einsumout_tensors = [torch.zeros((N, headNum, 512), dtype=data_type)]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        Einsumout_tensors_npu = [tensor.npu() for tensor in Einsumout_tensors]
        self.__set_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})
        self.mki.execute(in_tensors_npu, Einsumout_tensors_npu)
        self.__unset_envs({"ASDOPS_MATMUL_PP_FLAG": "1"})

        # Einsumout_tensors_npu = [torch.transpose(Einsumout_tensors_npu[0], 0, 1)]
        self.einsumOut = Einsumout_tensors_npu[0].cpu().clone()

        self.qOut_npu = torch.zeros((N, headNum, 576), dtype=data_type)
        OP_NAME = "RopeQConcatOperation"
        OP_PARAM0 = {}
        self.set_param(OP_NAME, OP_PARAM0)
        Split2_out_tensors_npu[1] = Split2_out_tensors_npu[1].cpu().reshape(N, headNum * 64)
        in_tensors = [Split2_out_tensors_npu[1], self.cos2, self.sin2, Einsumout_tensors_npu[0]]
        qOut_tensors = [self.qOut_npu]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        qout_tensors_npu = [qOut_tensors[i] if isinstance(i, int) else i.npu() for i in qOut_tensors]
        self.mki.execute(in_tensors_npu, qout_tensors_npu)
        qout_tensors_npu[0] = torch.cat(
            [qout_tensors_npu[0].cpu()[:, :, 64:], qout_tensors_npu[0].cpu()[:, :, :64]], dim=2
        )
        self.qOut_npu = qout_tensors_npu[0].cpu().clone()

        # cache mode 2 qOut nope quant
        qOutNopeInput = self.qOut_npu[:,:,0:512]
        scale = self.qNopeScale.cpu().clone()
        self.qOutNopeQuant = self.ein_sum_out_quant_golden(qOutNopeInput.clone(), scale)

        # #Rope
        OP_NAME = "RopeOperation"
        rotaryCoeff = 2
        OP_PARAM0 = {"rotaryCoeff": rotaryCoeff}
        self.RopeOut0 = torch.zeros((N, 64), dtype=data_type)
        self.RopeOut1 = torch.zeros((N, 64), dtype=data_type)
        seqlen = torch.randint(1, 2, (N, 1), dtype=torch.int32)
        in_tensors = [
            Split1_out_tensors_npu[1].reshape(N, 1 * 64),
            Split1_out_tensors_npu[1].reshape(N, 1 * 64),
            self.cos1,
            self.sin1,
            seqlen,
        ]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        Ropeout_tensors_npu = [self.RopeOut0.npu(), self.RopeOut1.npu()]
        self.set_param(OP_NAME, OP_PARAM0)
        self.mki.execute(in_tensors_npu, Ropeout_tensors_npu)
        keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)

        # RmsNorm3 bf16
        OP_NAME = "NormOperation"
        OP_PARAM0 = {"normType": 2, "inGamma": True, "epsilon": 1e-6, "precisionMode": 0, "gemmaMode": 0}

        in_tensors = [Split1_out_tensors_npu[0].reshape(N, 1, 512), self.gamma3]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        self.RmsNormOut = torch.zeros((N, 1, 512), dtype=data_type).npu()
        RmsNormOut_tensors_npu = [self.RmsNormOut]
        self.set_param(OP_NAME, OP_PARAM0)
        self.mki.execute(in_tensors_npu, RmsNormOut_tensors_npu)

        # kcache1, kcache2
        RmsNorm3Out = RmsNormOut_tensors_npu[0].cpu().clone()
        keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)

        if cacheMode == 2:
            #quant
            quantOut = self.quant(RmsNorm3Out, self.quantScale3)

            print(RmsNorm3Out.flatten()[2])
            nz = True
            # reshape
            if nz:
                self.keyCache1_out = self.reshapeAndCacheNz(quantOut, self.I8keyCache1, self.slotMapping, 512, 32, 16)
                keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)
                self.keyCache2_out = self.reshapeAndCacheNz(keyRope, self.keyCache2, self.slotMapping,64, 16, 4)
            else:
                self.keyCache1_out = self.reshapeAndCache(quantOut, self.I8keyCache1, self.slotMapping, 512)
                keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)
                self.keyCache2_out = self.reshapeAndCache(keyRope, self.keyCache2, self.slotMapping, 64)
        elif cacheMode == 3:
            print("================cacheMode:3")
            self.keyCache1_out = self.reshapeAndCacheNz(RmsNorm3Out, self.FPkeyCache1, self.slotMapping, 512, 16, 32)
            keyRope = Ropeout_tensors_npu[0].reshape(N, 1, 64)
            self.keyCache2_out = self.reshapeAndCacheNz(keyRope, self.keyCache2, self.slotMapping,64, 16, 4)
        else:
            out_tensors_npu = RmsNormOut_tensors_npu
            # Concat
            OP_NAME = "ConcatOperation"
            OP_PARAM = {"concatDim": 2}
            in_tensors = [RmsNormOut_tensors_npu[0], Ropeout_tensors_npu[0].cpu().reshape(N, 1, 64)]
            ConCat2out_tensors = [torch.zeros((N, 1, 576), dtype=data_type)]
            in_tensors_npu = [tensor.npu() for tensor in in_tensors]
            ConCat2out_tensors_npu = [tensor.npu() for tensor in ConCat2out_tensors]
            self.set_param(OP_NAME, OP_PARAM)
            self.mki.execute(in_tensors_npu, ConCat2out_tensors_npu)

            # Reshape&Cache
            OP_NAME = "ReshapeAndCacheOperation"
            OP_PARAM = {"type": 4}
            in_tensors = [ConCat2out_tensors_npu[0], self.keyOutTensor, self.slotMapping]
            out_tensors = [1]
            in_tensors_npu = [tensor.npu() for tensor in in_tensors]
            out_tensors_npu = [in_tensors_npu[i] if isinstance(i, int) else i.npu() for i in out_tensors]
            self.set_param(OP_NAME, OP_PARAM)
            self.mki.execute(in_tensors_npu, out_tensors_npu)
            self.keyout_npu = out_tensors_npu[0].cpu().clone()



    def reshapeAndCache(self, input, keycache, slotMapping,num):
        keycache = keycache.reshape(-1, num)
        input = input.reshape(-1,num)
        for i in range(len(slotMapping)):
            slot_idx = slotMapping[i]
            keycache[slot_idx] = input[i]
        return keycache

    def reshapeAndCacheNz(self, input, keycache, slotMapping,num,fenxin,loop):
        keycache = keycache.flatten()
        input = input.reshape(-1,num)
        for i in range(len(slotMapping)):
            slot_idx = slotMapping[i]
            outer_idx = (int)(slot_idx / 128)
            inner_idx = slot_idx % 128

            stride = 128*fenxin
            for j in range(loop):
                startIdx = inner_idx*fenxin + j*stride + outer_idx * 128 * num 
                keycache[startIdx: startIdx + fenxin] = input[i][j*fenxin : (j+1)*fenxin]

        return keycache


    def s8_saturation(self, inputdata):
        inputdata = torch.min(inputdata, torch.tensor(QUANTMAX, dtype=torch.float16))
        inputdata = torch.max(inputdata, torch.tensor(QUANTMIN, dtype=torch.float16))
        return np.rint(inputdata).to(torch.int8)

    def quant(self,x, qscale):
        # qscale = qscale.to(torch.float)
        qscale = 1 / qscale
        x = x.to(torch.float)
        # 使用广播机制来避免显式的循环
        scaled_values = (x * qscale).to(torch.float16)
        s8_res_cal = self.s8_saturation(scaled_values)
        return s8_res_cal


    def calc_vec_mm_data(self, N, headNum, data_type):
        mm1In = self.rms_norm_quant_calc(self.input1, self.gamma1, self.beta1, self.quantScale1, self.quantOffset1)

        self.rmsquantOut1 = mm1In.clone()
        mm1Out = torch.matmul(mm1In.to(torch.float32), self.wdqkv.transpose(0, 1).to(torch.float32))
        mm1Out = mm1Out.to(torch.int32) + self.bias1
        mm1Out = (mm1Out.to(torch.float32) * self.deScale1).to(data_type)
        self.mm1Out1 = mm1Out
        if data_type == torch.float16:
            self.deScale1 = process_deq_scale(deq_scale=self.deScale1)
        mm1OutSplit1, mm1OutSplit2 = torch.split(mm1Out, [576, 1536], dim=1)
        rms2Out = self.rms_norm_quant_calc(mm1OutSplit2, self.gamma2, self.beta2, self.quantScale2, self.quantOffset2)
        self.rmsquantOut2 = rms2Out.clone()
        mm2Out = torch.matmul(rms2Out.to(torch.float32), self.wuq.transpose(0, 1).to(torch.float32))
        mm2Out = mm2Out.to(torch.int32) + self.bias2
        mm2Out = (mm2Out.to(torch.float32) * self.deScale2).to(data_type)

        self.mm2Out2 = mm2Out
        if data_type == torch.float16:
            self.deScale2 = process_deq_scale(deq_scale=self.deScale2)

        mm11OutSplit1, mm12OutSplit1 = torch.split(mm1OutSplit1, [512, 64], dim=1)
        mm11OutSplit1 = mm11OutSplit1.reshape(N, 1, 512)
        self.keyOutTensor = self.keyCache.clone()
        self.keyOut1 = self.RmsNormAndRopeAndReshapeAndCacheGolden(
            mm11OutSplit1, self.gamma3, mm12OutSplit1, self.cos1, self.sin1, self.slotMapping, self.keyCache
        )
        mm2Out = mm2Out.reshape(N, headNum, 192)
        mm2OutSplit1, mm2OutSplit2 = torch.split(mm2Out, [128, 64], dim=2)
        self.bmmOut = torch.permute(
            torch.matmul(torch.permute(mm2Out[:, :, :128], (1, 0, 2)).float(), self.wuk.float()),
            (1, 0, 2),
        )
        self.gg = mm2OutSplit2.clone()
        qOut = self.RopeConcatGolden(mm2OutSplit2.reshape(N, headNum * 64), self.sin2, self.cos2, self.bmmOut)
        self.qOut = qOut.to(data_type)

    def golden_calc(self, in_tensors):
        return [self.qOut, self.keyOut1]

    def compare_data(self, tensor1, tensor2):
        out = tensor1.flatten()
        out_len = out.shape[0]
        golden = tensor2.flatten()
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        logging.info(f"maxDiff {max_diff}")
        golden = golden.to(torch.float32)
        out = out.to(torch.float32)
        limit_error = torch.maximum(torch.abs(golden * 0.001), torch.tensor(0.001))
        strict_limit_error = torch.maximum(torch.abs(golden * 0.003), torch.tensor(0.003))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        logging.info("1/1000 Accuracy is %f", 1 - float(error_count) / out_len)
        logging.info("3/1000 Accuracy is %f", 1 - float(strict_error_count) / out_len)
        logging.info("accuracy is correct: %r", (float(strict_error_count) / out_len) <= 0.001)

        return (float(strict_error_count) / out_len) <= 0.001

    def golden_compare(self, out_tensors, golden_out_tensors):
        logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if self.op_desc["specificParam"]["cacheMode"] == 0:
            result_double1 = compare_cv(self.qOut_npu, self.qOut, out_tensors[0])
            result_double2 = compare_cv(self.keyout_npu, self.keyOut1, out_tensors[1])
            logging.info("-------------------------------------------------------------------------------------------------------------------")
            return result_double1 and result_double2
        elif self.op_desc["specificParam"]["cacheMode"] == 1:
            result_double1 = compare_cv(self.qOut_npu[..., 0:512], self.qOut[..., 0:512], out_tensors[0])
            result_double2 = compare_cv(self.keyout_npu[..., 0:512], self.keyOut1[..., 0:512], out_tensors[1])
            result_double3 = compare_cv(self.qOut_npu[..., 512:576], self.qOut[..., 512:576], out_tensors[2])
            result_double4 = compare_cv(self.keyout_npu[..., 512:576], self.keyOut1[..., 512:576], out_tensors[3])
            logging.info("-------------------------------------------------------------------------------------------------------------------")
            return result_double1 and result_double2 and result_double3 and result_double4
        elif self.op_desc["specificParam"]["cacheMode"] == 2:
            diff = out_tensors[0].flatten() - self.qOutNopeQuant.flatten()
            max_diff = torch.max(torch.abs(diff))
            logging.info(f"-------- qNopeQuant max abs diff: {max_diff} -----------")
            result_double1 = max_diff <= 1
            result_double3 = compare_cv(self.qOut_npu[..., 512:576], self.qOut[..., 512:576], out_tensors[2])

            diff = self.keyCache1_out.flatten() - out_tensors[1].flatten()
            max_diff = torch.max(torch.abs(diff))
            logging.info(f"-------- KNopeQuant max abs diff: {max_diff} -----------")
            result_double2 = max_diff <= 1
            result_double4 = self.compare_data(self.keyCache2_out.flatten(), out_tensors[3].flatten())
            logging.info("-------------------------------------------------------------------------------------------------------------------")
            return result_double1 and result_double2 and result_double3 and result_double4
        elif self.op_desc["specificParam"]["cacheMode"] == 3:
            result_double1 = compare_cv(self.qOut_npu[..., 0:512], self.qOut[..., 0:512], out_tensors[0])
            result_double3 = compare_cv(self.qOut_npu[..., 512:576], self.qOut[..., 512:576], out_tensors[2])

            result_double2 = self.compare_data(self.keyCache1_out.flatten(), out_tensors[1].flatten())
            result_double4 = self.compare_data(self.keyCache2_out.flatten(), out_tensors[3].flatten())
            return result_double1 and result_double2 and result_double3 and result_double4

    #>>>>>>>>>>>>>>ut start
    @op_test.only_910b
    def test_mla_preprocess_fp16_cacheMode_0(self):
        N = 32
        headNum = 32
        cacheMode = 0
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 576), dtype=data_type), self.keyOutTensor, torch.tensor([]), torch.tensor([])],
        )

    @op_test.only_910b
    def test_mla_preprocess_fp16_cacheMode_1(self):
        N = 32
        headNum = 32
        cacheMode = 1
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [
                torch.zeros((N, headNum, 512), dtype=data_type),
                self.keyOutTensor[..., 0:512],
                torch.zeros((N, headNum, 64), dtype=data_type),
                self.keyOutTensor[..., 512:576],
            ],
        )

    @op_test.only_910b
    def test_mla_preprocess_fp16_cacheMode_2(self):
        N = 32
        headNum = 32
        cacheMode = 2
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                self.quantScale3,
                self.qNopeScale,
            ],
            [
                torch.zeros((N, headNum, 512), dtype=torch.int8),
                self.I8keyCache1, 
                torch.zeros((N, headNum, 64), dtype=data_type),
                self.keyCache2,
            ],
        )

    @op_test.only_910b
    def test_mla_preprocess_largeN_cacheMode_0(self):
        N = 1024
        headNum = 128
        cacheMode = 0
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 576), dtype=data_type), self.keyOutTensor, torch.tensor([]), torch.tensor([])],
        )

    @op_test.only_910b
    def test_mla_preprocess_largeN_cacheMode_1(self):
        N = 1024
        headNum = 128
        cacheMode = 1
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 512), dtype=data_type), self.keyOutTensor[..., 0:512], torch.zeros((N, headNum, 64), dtype=data_type), self.keyOutTensor[..., 512:576]],
        )

    @op_test.only_910b
    def test_mla_preprocess_largeN_cacheMode_2(self):
        N = 1024
        headNum = 128
        cacheMode = 2
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                self.quantScale3,
                self.qNopeScale,
            ],
            [
                torch.zeros((N, headNum, 512), dtype=torch.int8), 
                self.I8keyCache1,
                torch.zeros((N, headNum, 64), dtype=data_type), 
                self.keyCache2,
            ],
        )

    @op_test.only_910b
    def test_mla_preprocess_bf16_cacheMode_0(self):
        N = 32
        headNum = 32
        cacheMode = 0
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.bfloat16
        self.calc_vec_mm_atb_data_bf16(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1,
                self.deScale2,
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 576), dtype=data_type), self.keyOutTensor, torch.tensor([]), torch.tensor([])],   
        )

    @op_test.only_910b
    def test_mla_preprocess_bf16_cacheMode_1(self):
        N = 32
        headNum = 32
        cacheMode = 1
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.bfloat16
        self.calc_vec_mm_atb_data_bf16(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1,
                self.deScale2,
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 512), dtype=data_type), self.keyOutTensor[..., 0:512], torch.zeros((N, headNum, 64), dtype=data_type), self.keyOutTensor[..., 512:576]],   
        )

    @op_test.only_910b
    def test_mla_preprocess_bf16_cacheMode_2(self):
        N = 32
        headNum = 32
        cacheMode = 2
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.bfloat16
        self.calc_vec_mm_atb_data_bf16(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1,
                self.deScale2,
                self.quantScale3,
                self.qNopeScale,
            ],
            [
                torch.zeros((N, headNum, 512), dtype=torch.int8), 
                self.I8keyCache1,
                torch.zeros((N, headNum, 64), dtype=data_type), 
                self.keyCache2,
                ],   
        )

    @op_test.only_910b
    def test_mla_preprocess_bf16_largeN_cacheMode_0(self):
        N = 1024
        headNum = 32
        cacheMode = 0
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.bfloat16
        self.calc_vec_mm_atb_data_bf16(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1,
                self.deScale2,
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 576), dtype=data_type), self.keyOutTensor, torch.tensor([]), torch.tensor([])],
        )

    @op_test.only_910b
    def test_mla_preprocess_bf16_largeN_cacheMode_1(self):
        N = 1024
        headNum = 32
        cacheMode = 1
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.bfloat16
        self.calc_vec_mm_atb_data_bf16(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1,
                self.deScale2,
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 512), dtype=data_type), self.keyOutTensor[..., 0:512], torch.zeros((N, headNum, 64), dtype=data_type), self.keyOutTensor[..., 512:576]],
        )

    @op_test.only_910b
    def test_mla_preprocess_bf16_largeN_cacheMode_2(self):
        N = 1024
        headNum = 32
        cacheMode = 2
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.bfloat16
        self.calc_vec_mm_atb_data_bf16(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1,
                self.deScale2,
                self.quantScale3,
                self.qNopeScale,
            ],
            [
                torch.zeros((N, headNum, 512), dtype=torch.int8), 
                self.I8keyCache1,
                torch.zeros((N, headNum, 64), dtype=data_type), 
                self.keyCache2,
            ],
        )

    @op_test.only_910b
    def test_mla_preprocess_fp16_cacheMode_0_weight_nz(self):
        N = 32
        headNum = 32
        cacheMode = 0
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nz,  # self.wuk
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                transdata_3d(self.wuk, (16, 16)),  # self.wuk
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 576), dtype=data_type), self.keyOutTensor, torch.tensor([]), torch.tensor([])],
        )

    @op_test.only_910b
    def test_mla_preprocess_fp16_cacheMode_1_weight_nz(self):
        N = 32
        headNum = 32
        cacheMode = 1
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nz,  # self.wuk
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                transdata_3d(self.wuk, (16, 16)),  # self.wuk
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [
                torch.zeros((N, headNum, 512), dtype=data_type),
                self.keyOutTensor[..., 0:512],
                torch.zeros((N, headNum, 64), dtype=data_type),
                self.keyOutTensor[..., 512:576],
            ],
        )

    @op_test.only_910b
    def test_mla_preprocess_fp16_cacheMode_2_weight_nz(self):
        N = 32
        headNum = 32
        cacheMode = 2
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nz,  # self.wuk
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                transdata_3d(self.wuk, (16, 16)),  # self.wuk
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                self.quantScale3,
                self.qNopeScale,
            ],
            [
                torch.zeros((N, headNum, 512), dtype=torch.int8),
                self.I8keyCache1,
                torch.zeros((N, headNum, 64), dtype=data_type),
                self.keyCache2,
            ],
        )

    @op_test.only_910b
    def test_mla_preprocess_largeN_cacheMode_0_weight_nz(self):
        N = 1024
        headNum = 128
        cacheMode = 0
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nz,  # self.wuk
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                transdata_3d(self.wuk, (16, 16)),  # self.wuk
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 576), dtype=data_type), self.keyOutTensor, torch.tensor([]), torch.tensor([])],
        )

    @op_test.only_910b
    def test_mla_preprocess_largeN_cacheMode_1_weight_nz(self):
        N = 1024
        headNum = 128
        cacheMode = 1
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nz,  # self.wuk
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                transdata_3d(self.wuk, (16, 16)),  # self.wuk
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [
                torch.zeros((N, headNum, 512), dtype=data_type),
                self.keyOutTensor[..., 0:512],
                torch.zeros((N, headNum, 64), dtype=data_type),
                self.keyOutTensor[..., 512:576],
            ],
        )

    @op_test.only_910b
    def test_mla_preprocess_largeN_cacheMode_2_weight_nz(self):
        N = 1024
        headNum = 128
        cacheMode = 2
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nz,  # self.wuk
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                transdata_3d(self.wuk, (16, 16)),  # self.wuk
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                self.quantScale3,
                self.qNopeScale,
            ],
            [
                torch.zeros((N, headNum, 512), dtype=torch.int8),
                self.I8keyCache1,
                torch.zeros((N, headNum, 64), dtype=data_type),
                self.keyCache2,
            ],
        )
    
    @op_test.only_910b
    def test_mla_preprocess_fp16_gemv(self):
        N = 129
        headNum = 32
        cacheMode = 0
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 576), dtype=data_type), self.keyOutTensor, torch.tensor([]), torch.tensor([])],
        )

    @op_test.only_910b
    def test_mla_preprocess_bf16_gemv(self):
        N = 257
        headNum = 32
        cacheMode = 0
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.bfloat16
        self.calc_vec_mm_atb_data_bf16(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1,
                self.deScale2,
                torch.zeros((0), dtype=data_type),
                torch.zeros((0), dtype=data_type),
            ],
            [torch.zeros((N, headNum, 576), dtype=data_type), self.keyOutTensor, torch.tensor([]), torch.tensor([])],   
        )

    @op_test.only_910b
    def test_mla_preprocess_bf16_cacheMode_3(self):
        N = 1024
        headNum = 32
        cacheMode = 3
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.bfloat16
        self.calc_vec_mm_atb_data_bf16(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                self.wuk,
                self.deScale1,
                self.deScale2,
                self.quantScale3,
                self.qNopeScale,
            ],
            [
                torch.zeros((N, headNum, 512), dtype=data_type), 
                self.FPkeyCache1,
                torch.zeros((N, headNum, 64), dtype=data_type), 
                self.keyCache2
            ],
        )

    @op_test.only_910b
    def test_mla_preprocess_fp16_cacheMode_3(self):
        N = 32
        headNum = 32
        cacheMode = 3
        OP_PARAM = {"N": N, "headNum": headNum, "cacheMode": cacheMode}
        data_type = torch.float16
        self.calc_vec_mm_atb_data(N, headNum, data_type, cacheMode)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wdqkv,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nz,  # self.wuq,
                self.format_nd,
                self.format_nz,  # self.wuk
                self.format_nd,
                self.format_nd,
                self.format_nd,
                self.format_nd,
            ]
        )
        self.execute(
            [
                self.input1,
                self.gamma1,
                self.beta1,
                self.quantScale1,
                self.quantOffset1,
                transdata(self.wdqkv, (16, 32)),  # self.wdqkv,
                self.bias1,
                self.gamma2,
                self.beta2,
                self.quantScale2,
                self.quantOffset2,
                self.gamma3,
                self.sin1,
                self.cos1,
                self.sin2,
                self.cos2,
                self.keyCache,
                self.slotMapping,
                transdata(self.wuq, (16, 32)),  # self.wuq,
                self.bias2,
                transdata_3d(self.wuk, (16, 16)),  # self.wuk
                self.deScale1.to(torch.int64),
                self.deScale2.to(torch.int64),
                self.quantScale3,
                self.qNopeScale,
            ],
            [
                torch.zeros((N, headNum, 512), dtype=data_type),
                self.FPkeyCache1,
                torch.zeros((N, headNum, 64), dtype=data_type),
                self.keyCache2,
            ],
        )

if __name__ == "__main__":
    unittest.main()
