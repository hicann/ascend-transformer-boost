/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_MMA_H
#define INCLUDE_MMA_H

#include "hardware.h"
#include "kernel_tensor.h"

template <ArchType ArchTag, typename ElementA, typename ElementB, typename AccDTypeC, bool IsTransposeA>
struct mmad {
    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC) {};

    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC) {};
};

// Partial specialization for V220, int8_t, not_vector_A, not TransposeA
template <ArchType ArchTag, typename AccDTypeC, typename ElementA, typename ElementB>
struct mmad<ArchTag, ElementA, ElementB, AccDTypeC, false> {
    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC)
    {
#ifdef USE_ASCENDC
        AscendC::Mmad(l0cTensor,
                      l0aTensor,
                      l0bTensor,
                      AscendC::MmadParams(mTileActual, nTileActual, kPartActual, 0, false, initC));
#else
#if defined(__DAV_C220_CUBE__)
        mad((__cc__ AccDTypeC *)l0cTensor.GetPhyAddr(),
            (__ca__ ElementA *)l0aTensor.GetPhyAddr(),
            (__cb__ ElementB *)l0bTensor.GetPhyAddr(),
            mTileActual, // m
            kPartActual, // k
            nTileActual, // n
            0,           // unitFlag
            0,           // kDirectionAlign
            0,           // cmatrixSource
            initC        // cmatrixInitVal
        );
#else
        mad((__cc__ AccDTypeC *)l0cTensor.GetPhyAddr(),
            (__ca__ ElementA *)l0aTensor.GetPhyAddr(),
            (__cb__ ElementB *)l0bTensor.GetPhyAddr(),
            mTileActual, // m
            kPartActual, // k
            nTileActual, // n
            initC        // cmatrixInitVal
        );
#endif
#endif
    };

    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC)
    {
#ifdef USE_ASCENDC
        AscendC::LocalTensor<ElementA> biasTensor;
        biasTensor.InitBuffer(biasBt, mTileActual);
        biasTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
        AscendC::Mmad(l0cTensor,
                      l0aTensor,
                      l0bTensor,
                      biasTensor,
                      AscendC::MmadParams(mTileActual, nTileActual, kPartActual, 0, false, initC));
#else
        mad((__cc__ AccDTypeC *)l0cTensor.GetPhyAddr(),
            (__ca__ ElementA *)l0aTensor.GetPhyAddr(),
            (__cb__ ElementB *)l0bTensor.GetPhyAddr(),
            biasBt,
            mTileActual, // m
            kPartActual, // k
            nTileActual, // n
            0,           // unitFlag
            0,           // kDirectionAlign
            1,           // cmatrixSource
            0            // cmatrixInitVal
        );
#endif
    };
};

#endif
