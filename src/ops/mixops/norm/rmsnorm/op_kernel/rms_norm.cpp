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
#include "mixops/norm/common/common_tiling_data.h"

static constexpr uint32_t BUFFER_NUM = 1;       // split the UB to 2 equal part to enable ping-pong techniques.
static constexpr uint32_t DOUBLE_BUFFER = 2;     // split the UB to 2 equal part to enable ping-pong techniques.
static constexpr uint32_t BUF_FACTOR = 2;       // 1(g) + 1(workspace) = 2
static constexpr uint32_t OFFSET_GAMMA = 0;     // the offset of gamma is 0
static constexpr uint32_t OFFSET_WORKSPACE = 1; // the offset of workspace is 1
static constexpr uint32_t GEMMA_ON = 1;         // 开启gemmamode
static constexpr uint32_t GEMMA_OFF = 0;        // 关闭gemmamode
static constexpr uint32_t PRE_ON = 0;           // 开启precisionmode
static constexpr uint32_t PRE_OFF = 1;           // 关闭precisionmode，开始performancemode

using AscendC::HardEvent;

template<typename T, uint32_t gemmaMode, uint32_t precisionMode>
class RmsNormShort {
public:
    __aicore__ inline RmsNormShort(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *y,
                                   uint32_t num_core, uint32_t num_col, uint32_t num_row,
                                   float avg_factor, float epsilon, uint32_t slice_size)
        : num_core_(num_core), num_col_(num_col), avg_factor_(avg_factor),
          epsilon_(epsilon), slice_size_(slice_size)
    {
        uint32_t row_work = (num_row + num_core - 1) / num_core;
        if (AscendC::GetBlockIdx() != num_core - 1) {
            row_work_ = row_work;
        } else {
            row_work_ = num_row - (num_core - 1) * row_work;
        }
        gm_offset_ = static_cast<uint64_t>(row_work) * num_col_;
        gm_x_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        gm_g_.SetGlobalBuffer((__gm__ T *)g);
        gm_y_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gm_offset_);
        pipe.InitBuffer(fp16_x_que_, DOUBLE_BUFFER, num_col_ * sizeof(T));
        pipe.InitBuffer(fp16_g_que_, BUFFER_NUM, num_col_ * sizeof(T));
        pipe.InitBuffer(fp16_y_que_, BUFFER_NUM, num_col_ * sizeof(T));
        pipe.InitBuffer(fp32_g_buf_, num_col_ * sizeof(float));
        pipe.InitBuffer(fp32_xy_buf_, num_col_ * sizeof(float));
        pipe.InitBuffer(calc_buf_, BUF_FACTOR * num_col_ * sizeof(float));
#if __CCE_AICORE__ == 300
        pipe.InitBuffer(work_local_buf_, GetReduceSumWorkLocalSize<float>(num_col_) * sizeof(float));
#endif
    }

    __aicore__ inline void Launch()
    {
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        CopyInAndCastF32(fp32_xy, gm_x_, fp16_x_que_, 0, num_col_);
        CopyIn(gm_g_, fp16_g_que_, 0, num_col_);
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * num_col_];
#if __CCE_AICORE__ == 300
        AscendC::LocalTensor<float> work_local = work_local_buf_.Get<float>();
        float rms = sqrt(ComputeSliceSquareSum(fp32_xy, work, work_local, num_col_) * avg_factor_ + epsilon_);
#else
        float rms = sqrt(ComputeSliceSquareSum(fp32_xy, work, work, num_col_) * avg_factor_ + epsilon_);
#endif
        AscendC::LocalTensor<T> gamma = fp16_g_que_.DeQue<T>();
        AscendC::LocalTensor<float> fp32_g = fp32_g_buf_.Get<float>();
        CastGAndIsGemmaMode<T, gemmaMode>(fp32_g, gamma, num_col_);
        AscendC::LocalTensor<T> fp16_y = fp16_y_que_.AllocTensor<T>();
        ComputeRmsNormFast<T, precisionMode>(fp16_y, fp32_xy, rms, gamma, num_col_, work, fp32_g);
        fp16_y_que_.EnQue(fp16_y);
        CopyOut(gm_y_, fp16_y_que_, 0, num_col_);
        for (uint64_t pid = 1; pid < row_work_; pid++) {
            uint64_t offset = pid * num_col_;
            Compute(offset, gamma, fp32_g);
        }
        fp16_g_que_.FreeTensor(gamma);
    }

private:
    __aicore__ inline void Compute(uint64_t offset, const AscendC::LocalTensor<T> &gamma,
        const AscendC::LocalTensor<float> &fp32_g)
    {
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        CopyInAndCastF32(fp32_xy, gm_x_, fp16_x_que_, offset, num_col_);
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * num_col_];
#if __CCE_AICORE__ == 300
        AscendC::LocalTensor<float> work_local = work_local_buf_.Get<float>();
        float squareSum = ComputeSliceSquareSum(fp32_xy, work, work_local, num_col_);
#else
        float squareSum = ComputeSliceSquareSum(fp32_xy, work, work, num_col_);
#endif
        float rms = sqrt(squareSum * avg_factor_ + epsilon_);
        AscendC::LocalTensor<T> fp16_y = fp16_y_que_.AllocTensor<T>();
        ComputeRmsNormFast<T, precisionMode>(fp16_y, fp32_xy, rms, gamma, num_col_, work, fp32_g);
        fp16_y_que_.EnQue(fp16_y);
        CopyOut(gm_y_, fp16_y_que_, offset, num_col_);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, DOUBLE_BUFFER> fp16_x_que_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_g_que_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> fp16_y_que_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_g_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_;
#if __CCE_AICORE__ == 300
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_local_buf_;
#endif
    AscendC::GlobalTensor<T> gm_x_;
    AscendC::GlobalTensor<T> gm_g_;
    AscendC::GlobalTensor<T> gm_y_;
    uint32_t num_core_{0};  // 一共激活多少AICORE
    uint32_t num_col_{0};   // 输入的列数
    uint32_t row_work_{0};  // 需要计算多少行
    uint32_t row_step_{0};  // 除最后一次，每次搬入多少行
    uint32_t row_tail_{0};  // 最后一次搬入多少行数据
    uint64_t gm_offset_{0}; // GM数据起始位置偏移量
    uint32_t slice_size_{0};  // 每一行切分的大小
    float avg_factor_{1.0f}; // num_col_的倒数
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t precisionMode_{0};
    uint32_t gemmaMode_{0};
};

template<typename T>
class RmsNormLong {
public:
    __aicore__ inline RmsNormLong(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *y,
                                  uint32_t num_core, uint32_t num_col, uint32_t num_row,
                                  float avg_factor, float epsilon, uint32_t slice_size, uint32_t precisionMode,
                                  uint32_t gemmaMode)
        : num_core_(num_core), num_col_(num_col), avg_factor_(avg_factor),
          epsilon_(epsilon), slice_size_(slice_size), precisionMode_(precisionMode),
          gemmaMode_(gemmaMode)
    {
        uint32_t row_work = (num_row + num_core - 1) / num_core;
        if (AscendC::GetBlockIdx() != num_core - 1) {
            row_work_ = row_work;
        } else {
            row_work_ = num_row - (num_core - 1) * row_work;
        }
        num_slice_ = (num_col_ + slice_size_ - 1) / slice_size_;
        tail_size_ = num_col_ - (num_slice_ - 1) * slice_size_;
        gm_offset_ = static_cast<uint64_t>(row_work) * num_col_;

        gm_x_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        gm_g_.SetGlobalBuffer((__gm__ T *)g);
        gm_y_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gm_offset_);

        pipe.InitBuffer(fp16_x_que_, BUFFER_NUM, sizeof(T) * slice_size_);
        pipe.InitBuffer(fp16_g_que_, BUFFER_NUM, num_col_ * sizeof(T));
        pipe.InitBuffer(fp16_y_que_, BUFFER_NUM, sizeof(T) * slice_size_);
        pipe.InitBuffer(fp32_xy_buf_, sizeof(float) * slice_size_);
        pipe.InitBuffer(calc_buf_, BUF_FACTOR * sizeof(float) * slice_size_);
#if __CCE_AICORE__ == 300
        pipe.InitBuffer(work_local_buf_, GetReduceSumWorkLocalSize<float>(slice_size_) * sizeof(float));
#endif
    }

    __aicore__ inline void Launch()
    {
        for (uint64_t pid = 0; pid < row_work_; pid++) {
            uint64_t row_offset = pid * num_col_;
            float squareSum = 0.0f;
            for (uint32_t sid = 0; sid < num_slice_; sid++) {
                uint64_t col_offset = row_offset + sid * slice_size_;
                uint32_t eleNum = (sid == (num_slice_ - 1)) ? tail_size_ : slice_size_;
                squareSum += ComputeSquareSum(col_offset, eleNum);
            }
            float rms = sqrt(avg_factor_ * squareSum + epsilon_);
            for (uint64_t sid = 0; sid < num_slice_; sid++) {
                uint64_t sliceOffset = sid * slice_size_;
                uint64_t totalOffset = row_offset + sliceOffset;
                uint32_t eleNum = (sid == (num_slice_ - 1)) ? tail_size_ : slice_size_;
                ComputeNorm(rms, totalOffset, sliceOffset, eleNum);
            }
        }
    }

private:
    __aicore__ inline float ComputeSquareSum(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * slice_size_];

        CopyInAndCastF32(fp32_xy, gm_x_, fp16_x_que_, offset, numel);

#if __CCE_AICORE__ == 300
        AscendC::LocalTensor<float> work_local = work_local_buf_.Get<float>();
        return ComputeSliceSquareSum(fp32_xy, work, work_local, numel);
#else
        return ComputeSliceSquareSum(fp32_xy, work, work, numel);
#endif
    }

    __aicore__ inline void ComputeNorm(float rms, uint64_t totalOffset, uint64_t sliceOffset, uint32_t numel)
    {
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * slice_size_];
        AscendC::LocalTensor<T> fp16_y = fp16_y_que_.AllocTensor<T>();

        CopyInAndCastF32(fp32_xy, gm_x_, fp16_x_que_, totalOffset, numel);
        CopyIn(gm_g_, fp16_g_que_, sliceOffset, numel);

        AscendC::LocalTensor<T> gamma = fp16_g_que_.DeQue<T>();

        ComputeRmsNorm(fp16_y, fp32_xy, rms, gamma, numel, precisionMode_, gemmaMode_, work);

        fp16_g_que_.FreeTensor(gamma);

        fp16_y_que_.EnQue(fp16_y);
        CopyOut(gm_y_, fp16_y_que_, totalOffset, numel);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_x_que_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_g_que_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> fp16_y_que_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_;
#if __CCE_AICORE__ == 300
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_local_buf_;
#endif
    AscendC::GlobalTensor<T> gm_x_;
    AscendC::GlobalTensor<T> gm_g_;
    AscendC::GlobalTensor<T> gm_y_;
    uint32_t num_core_{0};  // 一共激活多少AICORE
    uint32_t num_col_{0};   // 输入的列数
    uint32_t row_work_{0};  // 需要计算多少行
    uint32_t row_step_{0};  // 除最后一次，每次搬入多少行
    uint32_t row_tail_{0};  // 最后一次搬入多少行数据
    uint64_t gm_offset_{0}; // GM数据起始位置偏移量
    uint32_t slice_size_{0};  // 每一行切分的大小
    int32_t num_slice_{0};
    int32_t tail_size_{0};
    float avg_factor_{1.0f}; // num_col_的倒数
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t precisionMode_{0};
    uint32_t gemmaMode_{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::RmsNormCommonTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 300)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ float *)(p_tilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ float *)(p_tilingdata + 16));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->mode = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->quantMin = (*(const __gm__ float *)(p_tilingdata + 28));
    tilingdata->precisionMode = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->gemmaMode = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::RmsNormCommonTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->avgFactor = (*(__ubuf__ float *)(tilingdata_in_ub + 12));
    tilingdata->epsilon = (*(__ubuf__ float *)(tilingdata_in_ub + 16));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->mode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->quantMin = (*(__ubuf__ float *)(tilingdata_in_ub + 28));
    tilingdata->precisionMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->gemmaMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                       \
    AtbOps::RmsNormCommonTilingData tiling_data;                                                                       \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void rms_norm(GM_ADDR x, GM_ADDR g, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(11100)) {  // fp16,gemmamode = 1,precisionmode = 0
        RmsNormShort<half, GEMMA_ON, PRE_ON> kernel(x, g, y, tiling_data.numCore, tiling_data.numCol,
                    tiling_data.numRow, tiling_data.avgFactor, tiling_data.epsilon, tiling_data.sliceSize);
        kernel.Launch();
    }
    if (TILING_KEY_IS(10100)){  // fp16,gemmode = 0,precisionmode = 0
        RmsNormShort<half, GEMMA_OFF, PRE_ON> kernel(x, g, y, tiling_data.numCore, tiling_data.numCol,
                    tiling_data.numRow, tiling_data.avgFactor, tiling_data.epsilon, tiling_data.sliceSize);
        kernel.Launch();
    }
    if (TILING_KEY_IS(10000)){  // fp16,gemmode = 0,precisionmode = 1, 启动高性能模式，高性能模式不支持BF16
        RmsNormShort<half, GEMMA_OFF, PRE_OFF> kernel(x, g, y, tiling_data.numCore, tiling_data.numCol,
                    tiling_data.numRow, tiling_data.avgFactor, tiling_data.epsilon, tiling_data.sliceSize);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(11110)){  // BF16,gemmamode = 1,precisionmode = 0
        RmsNormShort<bfloat16_t, GEMMA_ON, PRE_ON> kernel(x, g, y, tiling_data.numCore, tiling_data.numCol,
                    tiling_data.numRow, tiling_data.avgFactor, tiling_data.epsilon, tiling_data.sliceSize);
        kernel.Launch();
    }
    if (TILING_KEY_IS(10110)){  // BF16,gemmamode = 0,precisionmode = 0
        RmsNormShort<bfloat16_t, GEMMA_OFF, PRE_ON> kernel(x, g, y, tiling_data.numCore, tiling_data.numCol,
                    tiling_data.numRow, tiling_data.avgFactor, tiling_data.epsilon, tiling_data.sliceSize);
        kernel.Launch();
    }
#endif
    // Long tail
    if (TILING_KEY_IS(110000)) {
        RmsNormLong<half> kernel(x, g, y, tiling_data.numCore, tiling_data.numCol,
                                    tiling_data.numRow, tiling_data.avgFactor, tiling_data.epsilon,
                                    tiling_data.sliceSize, tiling_data.precisionMode, tiling_data.gemmaMode);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(110010)) {
        RmsNormLong<bfloat16_t> kernel(x, g, y, tiling_data.numCore, tiling_data.numCol,
                                        tiling_data.numRow, tiling_data.avgFactor, tiling_data.epsilon,
                                        tiling_data.sliceSize, tiling_data.precisionMode, tiling_data.gemmaMode);
        kernel.Launch();
    }
#endif
}
