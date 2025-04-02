/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "mixops/utils/common/kernel/kernel_utils.h"
#include "mixops/norm/postrmsnorm/tiling/post_rms_norm_tiling_data.h"
#include "mixops/norm/rmsnormforward/op_kernel/rms_norm_base.h"
#include "mixops/norm/common/common_pre_post/comm_pre_post.h"

using AscendC::HardEvent;

template<typename T, bool HAS_BIAS>
class PreRmsNormShort {
public:
    __aicore__ inline PreRmsNormShort(__gm__ uint8_t *x, __gm__ uint8_t *bias, __gm__ uint8_t *res_in,
                                      __gm__ uint8_t *g, __gm__ uint8_t *y, __gm__ uint8_t *res_out,
                                      AtbOps::PostRmsNormTilingData &tiling_data)
    {
        uint32_t numRow = tiling_data.numRow;
        numCore_ = tiling_data.numCore;
        numCol_ = tiling_data.numCol;
        avgFactor_ = *reinterpret_cast<float *>(&tiling_data.avgFactor);
        epsilon_ = *reinterpret_cast<float *>(&tiling_data.epsilon);
        sliceSize_ = tiling_data.sliceSize;
        precisionMode_ = tiling_data.precisionMode;
        uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;

        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore_ - 1) * rowWork;
        }
        gm_offset_ = static_cast<uint64_t>(rowWork) * numCol_;

        numColAlignFp32 = (numCol_ + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        numColAlignFp16 = (numCol_ + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

        gm_x_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        if constexpr (HAS_BIAS) {
            gm_bias_.SetGlobalBuffer((__gm__ T *)bias);
            pipe.InitBuffer(calc_buf_, NUM_TWO * sliceSize_ * sizeof(float));
            pipe.InitBuffer(fp16_x_buf_, NUM_FOUR * sliceSize_ * sizeof(T)); // x,res_in,gamma,bias
        } else {
            pipe.InitBuffer(calc_buf_, 1 * sliceSize_ * sizeof(float));
            pipe.InitBuffer(fp16_x_buf_, NUM_THREE * sliceSize_ * sizeof(T));
        }
        gm_g_.SetGlobalBuffer((__gm__ T *)g);
        gm_res_in_.SetGlobalBuffer((__gm__ T *)res_in + AscendC::GetBlockIdx() * gm_offset_);
        gm_y_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gm_offset_);
        gm_res_out_.SetGlobalBuffer((__gm__ T *)res_out + AscendC::GetBlockIdx() * gm_offset_);

        pipe.InitBuffer(fp32_xy_buf_, sliceSize_ * sizeof(float));
        pipe.InitBuffer(fp16_out_, sliceSize_ * sizeof(T));

        AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
        DataCopyCustom<T>(fp16_x, gm_x_[0], numColAlignFp16);
        DataCopyCustom<T>(fp16_x[sliceSize_], gm_res_in_[0], numColAlignFp16);
        DataCopy(fp16_x[sliceSize_ * NUM_TWO], gm_g_, numColAlignFp16);
    }

    __aicore__ inline void Launch()
    {
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
        AscendC::LocalTensor<T> fp16_res_in_ = fp16_x[sliceSize_];

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16_bias = fp16_x[sliceSize_ * NUM_THREE];
            AscendC::LocalTensor<float> fp32_bias = buf[sliceSize_];
            BiasIn(fp16_x, fp16_bias, fp32_bias, gm_bias_, numColAlignFp16);
        }

        uint64_t pid = 0;
        while (pid < rowWork_) {
            uint64_t offset = pid * numCol_;
            if (pid != 0) {
                CopyInXResIn(fp16_x, fp16_res_in_,
                    gm_x_, gm_res_in_, offset, numColAlignFp16);
            }
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            
            Compute(offset);

            AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyOut(offset, numCol_);
            AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            ++pid;
        }
    }

private:
    __aicore__ inline  void Compute(uint32_t offset)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
        AscendC::LocalTensor<T> fp16_res_in = fp16_x[sliceSize_];
        AscendC::LocalTensor<T> fp16_gamma = fp16_x[sliceSize_ * NUM_TWO];
        AscendC::LocalTensor<float> fp32_reduce_workspace = fp16_out_.Get<float>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<T> out_buf = fp16_out_.Get<T>();
        AscendC::LocalTensor<float> sqx = buf[0];

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16_bias = fp16_x[sliceSize_ * NUM_THREE];
            AscendC::LocalTensor<float> buf_bias = buf[sliceSize_];
            AddResBiasAndCast<T>(fp16_x, fp16_res_in, fp16_bias, fp32_xy, buf, buf_bias, numCol_);
        } else {
            AddResAndCast<T>(fp16_x, fp16_res_in, fp32_xy, buf, numCol_);
        }
        CastFrom32To16(fp16_x, fp32_xy, numCol_);
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

        DataCopyCustom<T>(gm_res_out_[offset], fp16_x, numCol_);
        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        FigureOutNorm(sqx, fp32_xy, fp32_reduce_workspace, avgFactor_, epsilon_, numCol_, numColAlignFp32);
        MultiplyGamma(fp16_gamma, sqx, fp32_xy, out_buf, numCol_, numColAlignFp32, numColAlignFp16, precisionMode_);
    }

    __aicore__ inline  void CopyOut(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> out_buf = fp16_out_.Get<T>();
        DataCopyCustom<T>(gm_y_[offset], out_buf, numel);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_x_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_y_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_res_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_, fp16_out_;
    AscendC::GlobalTensor<T> gm_x_;
    AscendC::GlobalTensor<T> gm_bias_;
    AscendC::GlobalTensor<T> gm_res_in_;
    AscendC::GlobalTensor<T> gm_g_;
    AscendC::GlobalTensor<T> gm_y_;
    AscendC::GlobalTensor<T> gm_res_out_;
    AtbOps::PostRmsNormTilingData tiling_data_;
    uint32_t numCore_{0};  // 一共激活多少AICORE
    uint32_t numCol_{0};   // 输入的列数
    uint32_t rowWork_{0};  // 需要计算多少行
    uint32_t rowStep_{0};  // 除最后一次，每次搬入多少行 rowTail_
    uint32_t rowTail_{0};  // 最后一次搬入多少行数据
    uint64_t gm_offset_{0}; // GM数据起始位置偏移量
    uint32_t sliceSize_{0};  // 每一行切分的大小
    float avgFactor_{1.0f}; // num_col_的倒数
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t numColAlignFp32{64};
    uint32_t numColAlignFp16{128};
    uint32_t precisionMode_{0};
};

template<typename T, bool HAS_BIAS>
class PreRmsNormLong {
public:
    __aicore__ inline PreRmsNormLong(__gm__ uint8_t *x, __gm__ uint8_t *bias,
                                     __gm__ uint8_t *res_in, __gm__ uint8_t *g,
                                     __gm__ uint8_t *y, __gm__ uint8_t *res_out,
                                     AtbOps::PostRmsNormTilingData &tiling_data)
    {
        uint32_t numRow = tiling_data.numRow;
        numCore_ = tiling_data.numCore;
        numCol_ = tiling_data.numCol;
        precisionMode_ = tiling_data.precisionMode;
        avgFactor_ = *reinterpret_cast<float *>(&tiling_data.avgFactor);
        epsilon_ = *reinterpret_cast<float *>(&tiling_data.epsilon);
        sliceSize_ = tiling_data.sliceSize;
        uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;
        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore_ - 1) * rowWork;
        }
#if __CCE_AICORE__ != 220
        if ((numCol_ % sliceSize_) * sizeof(T) < BLOCK_BYTE && (numCol_ % sliceSize_) != 0) {
            sliceSizeTmp_ = sliceSize_ - ((BLOCK_BYTE / sizeof(T)) - (numCol_ % sliceSize_));
        } else {
            sliceSizeTmp_ = sliceSize_;
        }
#else
        sliceSizeTmp_ = sliceSize_;
#endif
        numSlice_ = (numCol_ + sliceSizeTmp_ - 1) / sliceSizeTmp_;
        tailSize_ = numCol_ - (numSlice_ - 1) * sliceSizeTmp_;
        gm_offset_ = static_cast<uint64_t>(rowWork) * numCol_;
        gm_x_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        if constexpr (HAS_BIAS) {
            gm_bias_.SetGlobalBuffer((__gm__ T *)bias);
            pipe.InitBuffer(calc_buf_, NUM_TWO * sliceSize_ * sizeof(float));
            pipe.InitBuffer(fp16_x_buf_, NUM_FOUR * sliceSize_ * sizeof(T));

        } else {
            pipe.InitBuffer(calc_buf_, 1 * sliceSize_ * sizeof(float));
            pipe.InitBuffer(fp16_x_buf_, NUM_THREE * sliceSize_ * sizeof(T));
        }
        gm_res_in_.SetGlobalBuffer((__gm__ T *)res_in + AscendC::GetBlockIdx() * gm_offset_);
        gm_g_.SetGlobalBuffer((__gm__ T *)g);
        gm_y_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gm_offset_);
        gm_res_out_.SetGlobalBuffer((__gm__ T *)res_out + AscendC::GetBlockIdx() * gm_offset_);

        pipe.InitBuffer(fp32_xy_buf_, sliceSize_ * sizeof(float));
        pipe.InitBuffer(fp16_out_, sliceSize_ * sizeof(T));
    }

    __aicore__ inline void Launch()
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<T> fp16_res_in_ = fp16_x[sliceSize_];
        AscendC::LocalTensor<T> fp16_gamma = fp16_x[NUM_TWO * sliceSize_];
        uint64_t pid = 0;
        while (pid < rowWork_) {
            uint64_t rowOffset = pid * numCol_;
            uint32_t numEle = sliceSizeTmp_;
            squareSum_ = 0.0f;
            for (uint64_t sid = 0; sid < numSlice_; sid++) {
                uint64_t colOffset = rowOffset + sid * sliceSizeTmp_;
                if ((sid == (numSlice_ - 1)) && (tailSize_ != 0)) {
                    numEle = tailSize_;
                }
                numelAlignFp32 = (numEle + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
                numelAlignFp16 = (numEle + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

                CopyInXResIn(fp16_x, fp16_res_in_,
                    gm_x_, gm_res_in_, colOffset, numelAlignFp16);
                if constexpr (HAS_BIAS) {
                    CopyInBias(sid * sliceSizeTmp_);
                }
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                squareSum_ += ComputeSquareSum(numEle, sid, colOffset);
            }
            numEle = sliceSizeTmp_;
            float factor = avgFactor_ * squareSum_ + epsilon_;
            for (uint64_t sid = 0; sid < numSlice_; sid++) {
                uint64_t colOffset = rowOffset + sid * sliceSizeTmp_;
                if ((sid == (numSlice_ - 1)) && (tailSize_ != 0)) {
                    numEle = tailSize_;
                }
                numelAlignFp32 = (numEle + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
                numelAlignFp16 = (numEle + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;
                
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                CopyInXResIn(fp16_x, fp16_res_in_,
                    gm_x_, gm_res_in_, colOffset, numelAlignFp16);
                CopyInG(fp16_gamma, gm_g_, sid * sliceSizeTmp_, numelAlignFp16);
                if constexpr (HAS_BIAS) {
                    CopyInBias(sid * sliceSizeTmp_);
                }
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                ComputeNorm(factor, numEle);
                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);

                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyOut(colOffset, numEle);
                AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
            pid++;
        }
    }

private:

    __aicore__ inline void CopyInBias(uint64_t offset)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> fp32_bias = buf[sliceSize_];
        DataCopy(fp16_x[sliceSize_ * NUM_THREE], gm_bias_[offset], numelAlignFp16);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Cast(fp32_bias, fp16_x[sliceSize_ * NUM_THREE], AscendC::RoundMode::CAST_NONE, numelAlignFp16);
    }
    __aicore__ inline float ComputeSquareSum(uint32_t numel, uint32_t sid, uint64_t colOffset)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
        AscendC::LocalTensor<T> fp16_res_in = fp16_x[sliceSize_];
        AscendC::LocalTensor<float> fp32_reduce_workspace = fp16_out_.Get<float>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> sqx = buf[0];
        AscendC::LocalTensor<float> bias = buf[sliceSize_];

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16_bias = fp16_x[sliceSize_ * NUM_THREE];
            AscendC::LocalTensor<float> buf_bias = buf[sliceSize_];
            AddResBiasAndCast<T>(fp16_x, fp16_res_in, fp16_bias, fp32_xy, buf, buf_bias, numel);
        } else {
            AddResAndCast<T>(fp16_x, fp16_res_in, fp32_xy, buf, numel);
        }
        // fp32_xy = x + res_in
        CastFrom32To16(fp16_x, fp32_xy, numel);
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyCustom<T>(gm_res_out_[colOffset], fp16_x, numel);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        Mul(sqx, fp32_xy, fp32_xy, numelAlignFp32);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        ReduceSumCustom(sqx, sqx, fp32_reduce_workspace, numel);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return sqx.GetValue(0);
    }

    __aicore__ void ComputeNorm(float sqs, uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_buf_.Get<T>();
        AscendC::LocalTensor<T> fp16_res_in = fp16_x[sliceSize_];
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<T> out_buf = fp16_out_.Get<T>();
        AscendC::LocalTensor<float> sqx = buf[0];
        AscendC::LocalTensor<T> fp16_gamma = fp16_x[NUM_TWO * sliceSize_];

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16_bias = fp16_x[sliceSize_ * NUM_THREE];
            AscendC::LocalTensor<float> buf_bias = buf[sliceSize_];
            AddResBiasAndCast<T>(fp16_x, fp16_res_in, fp16_bias, fp32_xy, buf, buf_bias, numel);
        } else {
            AddResAndCast<T>(fp16_x, fp16_res_in, fp32_xy, buf, numel);
        }

        float factor = 1 / sqs;
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);

        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Duplicate(sqx, factor, AscendC::DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, AscendC::DEFAULT_REPEAT_STRIDE);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);

        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        factor = sqx.GetValue(0);
        Muls(fp32_xy, fp32_xy, factor, numelAlignFp32);
        AscendC::PipeBarrier<PIPE_V>();
        MultiplyGamma(fp16_gamma, sqx, fp32_xy,
            out_buf, numel, numelAlignFp32, numelAlignFp16, precisionMode_);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> out_buf = fp16_out_.Get<T>();
        DataCopyCustom<T>(gm_y_[offset], out_buf, numel);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_x_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_res_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_y_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_, fp16_out_;
    AscendC::GlobalTensor<T> gm_x_;
    AscendC::GlobalTensor<T> gm_bias_;
    AscendC::GlobalTensor<T> gm_g_;
    AscendC::GlobalTensor<T> gm_res_in_;
    AscendC::GlobalTensor<T> gm_y_;
    AscendC::GlobalTensor<T> gm_res_out_;
    uint32_t numCore_{0};  // 一共激活多少AICORE
    uint32_t numCol_{0};   // 输入的列数
    uint32_t rowStep_{0};  // 除最后一次，每次搬入多少行
    uint32_t rowWork_{0};  // 需要计算多少行
    uint32_t rowTail_{0};  // 最后一次搬入多少行数据
    uint64_t gm_offset_{0}; // GM数据起始位置偏移量
    uint32_t sliceSize_{0};  // 每一行切分的大小
    uint32_t sliceSizeTmp_{0};  // 每一行切分的大小
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t numSlice_{0};
    uint32_t tailSize_{0};
    float avgFactor_{1.0f}; // num_col_的倒数
    float squareSum_{0.0f};
    uint32_t numelAlignFp32{64};
    uint32_t numelAlignFp16{32};
    uint32_t precisionMode_{0};
};
inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::PostRmsNormTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->precisionMode = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::PostRmsNormTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->avgFactor = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->epsilon =  (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->precisionMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)  \
    AtbOps::PostRmsNormTilingData tiling_data;    \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void pre_rms_norm(GM_ADDR x, GM_ADDR bias, GM_ADDR res_in, GM_ADDR g, GM_ADDR y,
                                                   GM_ADDR res_out, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(0)) { // 000
        PreRmsNormShort<half, true> kernel(x, bias, res_in, g, y, res_out, tiling_data);
        kernel.Launch();
    } else if (TILING_KEY_IS(2)) { // 010
        PreRmsNormShort<half, false> kernel(x, bias, res_in, g, y, res_out, tiling_data);
        kernel.Launch();
    } else if (TILING_KEY_IS(4)) { // 100
        PreRmsNormLong<half, true> kernel(x, bias, res_in, g, y, res_out, tiling_data);
        kernel.Launch();
    } else if (TILING_KEY_IS(6)) { // 110
        PreRmsNormLong<half, false> kernel(x, bias, res_in, g, y, res_out, tiling_data);
        kernel.Launch();
    } else if (TILING_KEY_IS(1)) { // 001
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        PreRmsNormShort<bfloat16_t, true> kernel(x, bias, res_in, g, y, res_out, tiling_data);
        kernel.Launch();
#endif
    } else if (TILING_KEY_IS(3)) { // 011
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        PreRmsNormShort<bfloat16_t, false> kernel(x, bias, res_in, g, y, res_out, tiling_data);
        kernel.Launch();
#endif
    } else if (TILING_KEY_IS(5)) { // 101
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        PreRmsNormLong<bfloat16_t, true> kernel(x, bias, res_in, g, y, res_out, tiling_data);
        kernel.Launch();
#endif
    } else if (TILING_KEY_IS(7)) { // 111
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        PreRmsNormLong<bfloat16_t, false> kernel(x, bias, res_in, g, y, res_out, tiling_data);
        kernel.Launch();
#endif
    }
}