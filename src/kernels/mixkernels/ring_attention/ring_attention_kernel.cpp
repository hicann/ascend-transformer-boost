/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling/ring_attention_tiling.h"
#include "mixops/utils/common.h"


namespace AtbOps {
using namespace Mki;
constexpr uint64_t TENSOR_Q_SEQLEN_IDX = 6;
constexpr uint64_t TENSOR_KV_SEQLEN_IDX = 7;
constexpr uint32_t TILINGMIN = 512;
class RingAttentionBaseKernel : public KernelBase {
public:
    explicit RingAttentionBaseKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = Utils::RoundUp((TILING_PARA_SIZE + TILING_HEAD_SIZE) * sizeof(uint32_t), TILINGMIN);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RingAttention),
                  "unpad_flash_attention: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 12, "input num invalid", return false);
        auto &param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        auto dataShapeType = param.dataShapeType;
        if (dataShapeType == OpParam::RingAttention::DataShapeType::TYPE_BNSD) {
            // encoder shape [B,N,S,D], // decoder shape [numTokens,hiddenSize]
            MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2 ||
                          launchParam.GetInTensor(0).desc.dims.size() == 4,
                      "input 0 dim num invalid", return false);
            MKI_CHECK(param.quantType == OpParam::RingAttention::QuantType::TYPE_QUANT_UNDEFINED &&
                          !param.compressHead,
                      "BNSD is can not support quant,compressHead", return false);
        } else {
            MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3 ||
                          launchParam.GetInTensor(0).desc.dims.size() == 2,
                      "input 0 dim num invalid", return false);
        }

        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RingAttention),
                  "unpad_flash_attention: param type invalid", return false);
        auto &param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        auto batch = param.qSeqLen.size();
        MKI_CHECK(batch > 0 && batch <= ND_BATCH_LIMIT, "batch is invalid", return 0);
        uint64_t bufferSize =
            Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE * (batch - 1) * sizeof(uint32_t), TILINGMIN);
        return bufferSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        Status ret = RingAttentionTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(ret.Ok(), return ret);
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};


class RingAttentionKernel : public RingAttentionBaseKernel {
public:
    explicit RingAttentionKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : RingAttentionBaseKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RingAttention),
                  "unpad_flash_attention: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 14, "input num invalid", return false);
        auto &param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        auto dataShapeType = param.dataShapeType;
        if (dataShapeType == OpParam::RingAttention::DataShapeType::TYPE_BNSD) {
            // encoder shape [B,N,S,D], // decoder shape [numTokens,hiddenSize]
            MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2 ||
                          launchParam.GetInTensor(0).desc.dims.size() == 4,
                      "input 0 dim num invalid", return false);
            MKI_CHECK(param.quantType == OpParam::RingAttention::QuantType::TYPE_QUANT_UNDEFINED &&
                          !param.compressHead,
                      "BNSD is can not support quant,compressHead", return false);
        } else {
            MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3 ||
                          launchParam.GetInTensor(0).desc.dims.size() == 2,
                      "input 0 dim num invalid", return false);
        }

        MKI_CHECK(launchParam.GetOutTensorCount() == 2, "output num invalid", return false);
        return true;
    }
};


REG_KERNEL_BASE(RingAttentionKernel);
} // namespace AtbOps