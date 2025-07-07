/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/kernel_info.h>
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "mixkernels/pagedattention/tiling/paged_attention_910_95_tiling.h"
#include "mixkernels/pagedattention/tiling/paged_attention_tiling.h"
#include "mixkernels/pagedattention/tiling/paged_attention_tiling_dependency.h"
#include "mixkernels/utils/common.h"

namespace AtbOps {
using namespace Mki;

static const int TENSORLEN = 0;
static const int KVLISTLEN = 1;

class PagedAttentionAscend91095Kernel : public KernelBase {
public:
    explicit PagedAttentionAscend91095Kernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        auto &param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        SVector<int64_t> kvSeqLen;
        for (const auto &s : param.kvSeqLen) {
            kvSeqLen.push_back(static_cast<int64_t>(s));
        }
        return launchBufferSize_ + Utils::GetConstTensorSize<int64_t>(kvSeqLen);
    }

    Status Init(const LaunchParam &launchParam) override
    {
        return Status::OkStatus();
    }

    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return Status::OkStatus();
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return Status::OkStatus();
    }
};

class PagedAttentionBaseAscend91095Kernel : public PagedAttentionAscend91095Kernel {
public:
    explicit PagedAttentionBaseAscend91095Kernel(const std::string &kernelName, const BinHandle *handle)
        : PagedAttentionAscend91095Kernel(kernelName, handle)
    {
    }

    Status Init(const LaunchParam &launchParam) override
    {
        LaunchParam runLaunchParam;
        runLaunchParam.GetInTensors() = {launchParam.GetInTensor(0),
                                         launchParam.GetInTensor(1),
                                         launchParam.GetInTensor(2),
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         launchParam.GetInTensor(3),
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {}};
        runLaunchParam.GetOutTensors() = {launchParam.GetOutTensor(0), launchParam.GetOutTensor(0)};
        auto &param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        runLaunchParam.SetParam(param);
        SVector<int> tempInputLens = {TENSORLEN, KVLISTLEN, KVLISTLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN};
        runLaunchParam.SetInputLens(tempInputLens);
        return KernelBase::Init(runLaunchParam);
    }

    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        LaunchParam runLaunchParam;
        runLaunchParam.GetInTensors() = {{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                                         {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
        runLaunchParam.GetOutTensors() = {{}, {}};
        runLaunchParam.GetInTensor(0).data = launchParam.GetInTensor(0).data;
        runLaunchParam.GetInTensor(1).data = launchParam.GetInTensor(1).data;
        runLaunchParam.GetInTensor(2).data = launchParam.GetInTensor(2).data;
        runLaunchParam.GetInTensor(13).data = launchParam.GetInTensor(3).data;
        runLaunchParam.GetOutTensor(0).data = launchParam.GetOutTensor(0).data;
        runLaunchParam.GetOutTensor(1).data = launchParam.GetOutTensor(0).data;
        SVector<int> tempInputLens = {TENSORLEN, KVLISTLEN, KVLISTLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN};
        runLaunchParam.SetInputLens(tempInputLens);
        return KernelBase::Run(runLaunchParam, runInfo);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto ret = PagedAttentionBaseAscend91095Tiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(ret.Ok(), return ret);
        // When in a non-emulator environment, add "kernelInfo_.SetHwsyncIdx(0);" here.
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        SVector<int64_t> kvSeqLen;
        for (const auto &s : param.kvSeqLen) {
            kvSeqLen.push_back(static_cast<int64_t>(s));
        }
        kernelInfo_.AddConstTensorData<int64_t>(6, kvSeqLen);
        return Status::OkStatus();
    }
};

class PagedAttentionDequantFusionAscend91095Kernel : public PagedAttentionAscend91095Kernel {
public:
    explicit PagedAttentionDequantFusionAscend91095Kernel(const std::string &kernelName, const BinHandle *handle)
        : PagedAttentionAscend91095Kernel(kernelName, handle)
    {
    }

    Status Init(const LaunchParam &launchParam) override
    {
        LaunchParam runLaunchParam;
        runLaunchParam.GetInTensors() = {launchParam.GetInTensor(0),
                                         launchParam.GetInTensor(1),
                                         launchParam.GetInTensor(2),
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         launchParam.GetInTensor(3),
                                         {},
                                         {},
                                         launchParam.GetInTensor(5),
                                         {},
                                         launchParam.GetInTensor(7),
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {}};
        runLaunchParam.GetOutTensors() = {launchParam.GetOutTensor(0), launchParam.GetOutTensor(0)};
        auto &param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        runLaunchParam.SetParam(param);
        SVector<int> tempInputLens = {TENSORLEN, KVLISTLEN, KVLISTLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN};
        runLaunchParam.SetInputLens(tempInputLens);
        return KernelBase::Init(runLaunchParam);
    }

    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        LaunchParam runLaunchParam;
        runLaunchParam.GetInTensors() = {{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                                         {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
        runLaunchParam.GetOutTensors() = {{}, {}};
        runLaunchParam.GetInTensor(0).data = launchParam.GetInTensor(0).data;
        runLaunchParam.GetInTensor(1).data = launchParam.GetInTensor(1).data;
        runLaunchParam.GetInTensor(2).data = launchParam.GetInTensor(2).data;
        runLaunchParam.GetInTensor(13).data = launchParam.GetInTensor(3).data;
        runLaunchParam.GetInTensor(16).data = launchParam.GetInTensor(5).data;
        runLaunchParam.GetInTensor(18).data = launchParam.GetInTensor(7).data;
        runLaunchParam.GetOutTensor(0).data = launchParam.GetOutTensor(0).data;
        runLaunchParam.GetOutTensor(1).data = launchParam.GetOutTensor(0).data;
        SVector<int> tempInputLens = {TENSORLEN, KVLISTLEN, KVLISTLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN,
                                      TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN, TENSORLEN};
        runLaunchParam.SetInputLens(tempInputLens);
        return KernelBase::Run(runLaunchParam, runInfo);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto ret = PagedAttentionW8A16Ascend91095Tiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(ret.Ok(), return ret);
        // When in a non-emulator environment, add "kernelInfo_.SetHwsyncIdx(0);" here.
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        SVector<int64_t> kvSeqLen;
        for (const auto &s : param.kvSeqLen) {
            kvSeqLen.push_back(static_cast<int64_t>(s));
        }
        kernelInfo_.AddConstTensorData<int64_t>(6, kvSeqLen);
        return Status::OkStatus();
    }
};

class PagedAttentionQFloat16KvFloat16OutFloat16Ascend91095Kernel : public PagedAttentionBaseAscend91095Kernel {
public:
    explicit PagedAttentionQFloat16KvFloat16OutFloat16Ascend91095Kernel(const std::string &kernelName,
                                                                        const BinHandle *handle)
        : PagedAttentionBaseAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQBf16KvBf16OutBf16Ascend91095Kernel : public PagedAttentionBaseAscend91095Kernel {
public:
    explicit PagedAttentionQBf16KvBf16OutBf16Ascend91095Kernel(const std::string &kernelName, const BinHandle *handle)
        : PagedAttentionBaseAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQFloat16KvFloat8e5m2OutFloat16Ascend91095Kernel
    : public PagedAttentionDequantFusionAscend91095Kernel {
public:
    explicit PagedAttentionQFloat16KvFloat8e5m2OutFloat16Ascend91095Kernel(const std::string &kernelName,
                                                                           const BinHandle *handle)
        : PagedAttentionDequantFusionAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQBf16KvFloat8e5m2OutBf16Ascend91095Kernel : public PagedAttentionDequantFusionAscend91095Kernel {
public:
    explicit PagedAttentionQBf16KvFloat8e5m2OutBf16Ascend91095Kernel(const std::string &kernelName,
                                                                     const BinHandle *handle)
        : PagedAttentionDequantFusionAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQFloat16KvFloat8e4m3fnOutFloat16Ascend91095Kernel
    : public PagedAttentionDequantFusionAscend91095Kernel {
public:
    explicit PagedAttentionQFloat16KvFloat8e4m3fnOutFloat16Ascend91095Kernel(const std::string &kernelName,
                                                                             const BinHandle *handle)
        : PagedAttentionDequantFusionAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQBf16KvFloat8e4m3fnOutBf16Ascend91095Kernel : public PagedAttentionDequantFusionAscend91095Kernel {
public:
    explicit PagedAttentionQBf16KvFloat8e4m3fnOutBf16Ascend91095Kernel(const std::string &kernelName,
                                                                       const BinHandle *handle)
        : PagedAttentionDequantFusionAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQFloat16KvHifloat8OutFloat16Ascend91095Kernel
    : public PagedAttentionDequantFusionAscend91095Kernel {
public:
    explicit PagedAttentionQFloat16KvHifloat8OutFloat16Ascend91095Kernel(const std::string &kernelName,
                                                                         const BinHandle *handle)
        : PagedAttentionDequantFusionAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQBf16KvHifloat8OutBf16Ascend91095Kernel : public PagedAttentionDequantFusionAscend91095Kernel {
public:
    explicit PagedAttentionQBf16KvHifloat8OutBf16Ascend91095Kernel(const std::string &kernelName,
                                                                   const BinHandle *handle)
        : PagedAttentionDequantFusionAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQFloat16KvInt8OutFloat16Ascend91095Kernel : public PagedAttentionDequantFusionAscend91095Kernel {
public:
    explicit PagedAttentionQFloat16KvInt8OutFloat16Ascend91095Kernel(const std::string &kernelName,
                                                                     const BinHandle *handle)
        : PagedAttentionDequantFusionAscend91095Kernel(kernelName, handle)
    {
    }
};

class PagedAttentionQBf16KvInt8OutBf16Ascend91095Kernel : public PagedAttentionDequantFusionAscend91095Kernel {
public:
    explicit PagedAttentionQBf16KvInt8OutBf16Ascend91095Kernel(const std::string &kernelName, const BinHandle *handle)
        : PagedAttentionDequantFusionAscend91095Kernel(kernelName, handle)
    {
    }
};

REG_KERNEL_BASE(PagedAttentionQFloat16KvFloat16OutFloat16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQBf16KvBf16OutBf16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQFloat16KvFloat8e5m2OutFloat16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQBf16KvFloat8e5m2OutBf16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQFloat16KvFloat8e4m3fnOutFloat16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQBf16KvFloat8e4m3fnOutBf16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQFloat16KvHifloat8OutFloat16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQBf16KvHifloat8OutBf16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQFloat16KvInt8OutFloat16Ascend91095Kernel);
REG_KERNEL_BASE(PagedAttentionQBf16KvInt8OutBf16Ascend91095Kernel);

} // namespace AtbOps
