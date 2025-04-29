/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_UTILS_H
#define INCLUDE_UTILS_H

template <typename IN_DTYPE>
__aicore__ inline void CreateCaMatrix(const AscendC::LocalTensor<IN_DTYPE> &dst,
                                      const uint16_t repeats,
                                      const uint16_t blockNum,
                                      const uint16_t dstGap,
                                      const IN_DTYPE initValue)
{
#ifdef USE_ASCENDC
    AscendC::InitConstValue<IN_DTYPE>(dst,
                                      AscendC::InitConstValueParams<IN_DTYPE>(repeats, blockNum, dstGap, initValue));
#else
    uint64_t repeatConfig = ((uint64_t)blockNum << 16) | ((uint64_t)dstGap << 32) | (uint64_t)repeats;
    create_ca_matrix((__ca__ IN_DTYPE *)dst.GetPhyAddr(), repeatConfig, initValue);
#endif
}
__aicore__ inline void SetFftsBaseAddr(uint64_t config)
{
#ifdef USE_ASCENDC
    AscendC::SetSyncBaseAddr(config);
#else
    set_ffts_base_addr(config);
#endif
}
template <typename IN_DTYPE>
__aicore__ inline void SetPadding(IN_DTYPE padValue)
{
#ifdef USE_ASCENDC
    AscendC::SetLoadDataPaddingValue<IN_DTYPE>(padValue);
#else
    set_padding(padValue);
#endif
}
__aicore__ inline void SetAtomicnone()
{
#ifdef USE_ASCENDC
    AscendC::SetAtomicNone();
#else
    set_atomic_none();
#endif
}
__aicore__ inline void SetMasknorm()
{
#if __CCE_AICORE__ == 100
    return;
#endif
#ifdef USE_ASCENDC
    AscendC::SetMaskNorm();
#else
    set_mask_norm();
#endif
}
__aicore__ inline void SetNdpara(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
#ifdef USE_ASCENDC
    AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride);
#else
    uint64_t config = (uint64_t)ndNum | ((uint64_t)srcNdStride << 16) | ((uint64_t)dstNdStride << 32);
    set_nd_para(config);
#endif
}
template <typename IN_DTYPE>
__aicore__ inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow)
{
#ifdef USE_ASCENDC
    AscendC::SetVectorMask<IN_DTYPE>(maskHigh, maskLow);
#else
    set_vector_mask(maskHigh, maskLow);
#endif
}
__aicore__ inline int64_t GetSubBlockidx()
{
#ifdef USE_ASCENDC
    return AscendC::GetSubBlockIdx();
#else
    return get_subblockid();
#endif
}
__aicore__ inline void WaitFlagDev(uint16_t flagId)
{
#ifdef USE_ASCENDC
    AscendC::WaitEvent(flagId);
#else
    wait_flag_dev(flagId);
#endif
}
template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
#ifdef USE_ASCENDC
    AscendC::CrossCoreSetFlag<mode, pipe>(flagId);
#else
    uint64_t config = (0x1 | ((uint64_t)(mode & 0x3) << 4) | (uint64_t)(flagId & 0xf) << 8);
    ffts_cross_core_sync(pipe, config);
#endif
}
template <typename IN_DTYPE, bool setRelu = false>
__aicore__ inline void SetFpc(const AscendC::LocalTensor<IN_DTYPE> &preTensor, bool isUnitFlag = false)
{
#ifdef USE_ASCENDC
    AscendC::SetFixPipeConfig<IN_DTYPE, setRelu>(preTensor, isUnitFlag);
#else
    uint64_t config = (((uint64_t)preTensor.GetPhyAddr() >> 7) << 8) | ((uint64_t)isUnitFlag << 63);
    set_fpc(config);
#endif
}
template <typename IN_DTYPE>
__aicore__ inline void CopyCbufToFbuf(AscendC::LocalTensor<IN_DTYPE> &dst,
                                      AscendC::LocalTensor<IN_DTYPE> &src,
                                      uint16_t burstNum,
                                      uint16_t burstLen,
                                      uint16_t srcGapSize,
                                      uint16_t dstGapSize)
{
#ifdef USE_ASCENDC
    dst.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2PIPE2GM);
    AscendC::DataCopy(dst,
                      src,
                      AscendC::DataCopyParams(burstNum,     // nBurst
                                              burstLen,     // lenBurst
                                              srcGapSize,   // srcGap
                                              dstGapSize)); // dstGap);
#else
    copy_cbuf_to_fbuf(((__fbuf__ IN_DTYPE *)dst.GetPhyAddr()),
                      ((__cbuf__ IN_DTYPE *)src.GetPhyAddr()),
                      burstNum,
                      burstLen,
                      srcGapSize,
                      dstGapSize);
#endif
}
template <typename IN_DTYPE>
__aicore__ inline void CopyCbufToBt(uint64_t dst,
                                    const AscendC::LocalTensor<IN_DTYPE> &src,
                                    uint16_t convControl,
                                    uint16_t nBurst,
                                    uint16_t lenBurst,
                                    uint16_t sourceGap,
                                    uint16_t dstGap)
{
#ifdef USE_ASCENDC
    AscendC::LocalTensor<IN_DTYPE> dstTensor;
    dstTensor.InitBuffer(dst, nBurst * lenBurst);
    dstTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
    AscendC::DataCopy(dstTensor,
                      src,
                      AscendC::DataCopyParams(nBurst,    // nBurst
                                              lenBurst,  // lenBurst
                                              sourceGap, // srcGap
                                              dstGap));  // dstGap);
#else
    copy_cbuf_to_bt(dst, ((__cbuf__ IN_DTYPE *)src.GetPhyAddr()), convControl, nBurst, lenBurst, sourceGap, dstGap);
#endif
}
#endif