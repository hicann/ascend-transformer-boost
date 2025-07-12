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
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
#include "mixkernels/utils/common.h"
#include "tiling/reshape_and_cache_tiling.h"
#include "tiling/reshape_and_cache_95_tiling.h"
#include "tiling/reshape_and_cache_tiling_dependency.h"
#include <mki/utils/platform/platform_info.h>

namespace AtbOps {
using namespace Mki;
class ReshapeAndCacheKernel : public KernelBase {
public:
    explicit ReshapeAndCacheKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::ReshapeAndCache>(launchParam.GetParam());

        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ReshapeAndCache),
            "reshape_and_cache: param type invalid", return false);
        if (param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND ||
            param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ ||
            param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS ||
            param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE ||
            param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_OMNI_COMPRESS) {
            MKI_CHECK(launchParam.GetOutTensorCount() == DIM_2, "output num invalid", return false);
        }
        if (param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO) {
            MKI_CHECK(launchParam.GetOutTensorCount() == DIM_1, "output num invalid", return false);
        }
        
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_3, "tensor 0 dim num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        return TILING_PARA_SIZE; // 64 is tiling buffer size
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = ReshapeAndCacheTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        return Status::OkStatus();
    }
};

class ReshapeAndCacheNdKernel : public ReshapeAndCacheKernel {
public:
    explicit ReshapeAndCacheNdKernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheKernel(kernelName, handle)
    {
    }
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReshapeAndCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 5, "input num invalid", return false);
        return true;
    }
};

class ReshapeAndCacheCompressKernel : public ReshapeAndCacheKernel {
public:
    explicit ReshapeAndCacheCompressKernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheKernel(kernelName, handle)
    {
    }
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReshapeAndCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 7, "input num invalid", return false);
        return true;
    }
};

class ReshapeAndCacheCompressRopeKernel : public ReshapeAndCacheKernel {
public:
    explicit ReshapeAndCacheCompressRopeKernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheKernel(kernelName, handle)
    {
    }
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReshapeAndCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 8, "input num invalid", return false);
        return true;
    }
};

class ReshapeAndCacheNzKernel : public ReshapeAndCacheKernel {
public:
    explicit ReshapeAndCacheNzKernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheKernel(kernelName, handle)
    {
    }
        bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReshapeAndCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 5, "input num invalid", return false);
        return true;
    }
};

class ReshapeAndCacheNdSisoKernel : public ReshapeAndCacheKernel {
public:
    explicit ReshapeAndCacheNdSisoKernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheKernel(kernelName, handle)
    {
    }
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReshapeAndCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        return true;
    }
};

class ReshapeAndCacheOmniCompressKernel : public ReshapeAndCacheKernel {
public:
    explicit ReshapeAndCacheOmniCompressKernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheKernel(kernelName, handle)
    {
    }
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReshapeAndCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 8, "input num invalid", return false);
        return true;
    }
};

class ReshapeAndCache95Kernel : public KernelBase {
public:
    explicit ReshapeAndCache95Kernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = AtbOps::ReshapeAndCache95Tiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        
        return Status::OkStatus();
    }

    Status Init(const LaunchParam &launchParam) override
    {
        LaunchParam newLaunchParam = launchParam;
        newLaunchParam.GetInTensors() = {launchParam.GetInTensor(0), launchParam.GetInTensor(2),
        launchParam.GetInTensor(4), launchParam.GetInTensor(1), launchParam.GetInTensor(3)};
        newLaunchParam.AddInTensor({});
        newLaunchParam.AddInTensor({});
        newLaunchParam.AddInTensor({});
        
        return KernelBase::Init(newLaunchParam);
    }

    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        LaunchParam newLaunchParam = launchParam;
        newLaunchParam.GetInTensors() = {launchParam.GetInTensor(0), launchParam.GetInTensor(2),
        launchParam.GetInTensor(4), launchParam.GetInTensor(1), launchParam.GetInTensor(3)};
        newLaunchParam.AddInTensor({});
        newLaunchParam.AddInTensor({});
        newLaunchParam.AddInTensor({});

        return KernelBase::Run(newLaunchParam, runInfo);
    }
};

class ReshapeAndCacheF16Nd95Kernel : public ReshapeAndCache95Kernel {
public:
    explicit ReshapeAndCacheF16Nd95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCache95Kernel(kernelName, handle)
    {
    }
};

class ReshapeAndCacheBF16Nd95Kernel : public ReshapeAndCache95Kernel {
public:
    explicit ReshapeAndCacheBF16Nd95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCache95Kernel(kernelName, handle)
    {
    }
};

class ReshapeAndCacheINT8Nd95Kernel : public ReshapeAndCache95Kernel {
public:
    explicit ReshapeAndCacheINT8Nd95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCache95Kernel(kernelName, handle)
    {
    }
};

class ReshapeAndCacheF8e4m3Nd95Kernel : public ReshapeAndCache95Kernel {
public:
    explicit ReshapeAndCacheF8e4m3Nd95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCache95Kernel(kernelName, handle)
    {
    }
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return Status::OkStatus();
    }
};

class ReshapeAndCacheF8e5m2Nd95Kernel : public ReshapeAndCache95Kernel {
public:
    explicit ReshapeAndCacheF8e5m2Nd95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCache95Kernel(kernelName, handle)
    {
    }
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return Status::OkStatus();
    }
};

class ReshapeAndCacheCompressAlibi95Kernel : public KernelBase {
public:
    explicit ReshapeAndCacheCompressAlibi95Kernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = AtbOps::ReshapeAndCacheCompressAlibi95Tiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        
        return Status::OkStatus();
    }

    Status Init(const LaunchParam &launchParam) override
    {
        LaunchParam newLaunchParam = launchParam;
        newLaunchParam.GetInTensors() = {launchParam.GetInTensor(0), launchParam.GetInTensor(2),
        launchParam.GetInTensor(4), launchParam.GetInTensor(1), launchParam.GetInTensor(3), launchParam.GetInTensor(5),
        {}, launchParam.GetInTensor(6)};
        
        return KernelBase::Init(newLaunchParam);
    }

    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        LaunchParam newLaunchParam = launchParam;
        newLaunchParam.GetInTensors() = {launchParam.GetInTensor(0), launchParam.GetInTensor(2),
        launchParam.GetInTensor(4), launchParam.GetInTensor(1), launchParam.GetInTensor(3), launchParam.GetInTensor(5),
        {}, launchParam.GetInTensor(6)};

        return KernelBase::Run(newLaunchParam, runInfo);
    }
};

class ReshapeAndCacheF16CompressAlibi95Kernel : public ReshapeAndCacheCompressAlibi95Kernel {
public:
    explicit ReshapeAndCacheF16CompressAlibi95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheCompressAlibi95Kernel(kernelName, handle)
    {
    }
};

class ReshapeAndCacheBF16CompressAlibi95Kernel : public ReshapeAndCacheCompressAlibi95Kernel {
public:
    explicit ReshapeAndCacheBF16CompressAlibi95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheCompressAlibi95Kernel(kernelName, handle)
    {
    }
};

class ReshapeAndCacheINT8CompressAlibi95Kernel : public ReshapeAndCacheCompressAlibi95Kernel {
public:
    explicit ReshapeAndCacheINT8CompressAlibi95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheCompressAlibi95Kernel(kernelName, handle)
    {
    }
};

class ReshapeAndCacheCompressRope95Kernel : public KernelBase {
public:
    explicit ReshapeAndCacheCompressRope95Kernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = AtbOps::ReshapeAndCacheCompressAlibi95Tiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        
        return Status::OkStatus();
    }

    Status Init(const LaunchParam &launchParam) override
    {
        LaunchParam newLaunchParam = launchParam;
        newLaunchParam.GetInTensors() = {launchParam.GetInTensor(0), launchParam.GetInTensor(2),
        launchParam.GetInTensor(4), launchParam.GetInTensor(1), launchParam.GetInTensor(3), launchParam.GetInTensor(5),
        launchParam.GetInTensor(7), launchParam.GetInTensor(6)};
        
        return KernelBase::Init(newLaunchParam);
    }

    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        LaunchParam newLaunchParam = launchParam;
        newLaunchParam.GetInTensors() = {launchParam.GetInTensor(0), launchParam.GetInTensor(2),
        launchParam.GetInTensor(4), launchParam.GetInTensor(1), launchParam.GetInTensor(3), launchParam.GetInTensor(5),
        launchParam.GetInTensor(7), launchParam.GetInTensor(6)};

        return KernelBase::Run(newLaunchParam, runInfo);
    }
};

class ReshapeAndCacheF16CompressRope95Kernel : public ReshapeAndCacheCompressRope95Kernel {
public:
    explicit ReshapeAndCacheF16CompressRope95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheCompressRope95Kernel(kernelName, handle)
    {
    }
};

class ReshapeAndCacheBF16CompressRope95Kernel : public ReshapeAndCacheCompressRope95Kernel {
public:
    explicit ReshapeAndCacheBF16CompressRope95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheCompressRope95Kernel(kernelName, handle)
    {
    }
};

class ReshapeAndCacheINT8CompressRope95Kernel : public ReshapeAndCacheCompressRope95Kernel {
public:
    explicit ReshapeAndCacheINT8CompressRope95Kernel(const std::string &kernelName, const BinHandle *handle)
        : ReshapeAndCacheCompressRope95Kernel(kernelName, handle)
    {
    }
};

REG_KERNEL_BASE(ReshapeAndCacheNdKernel);
REG_KERNEL_BASE(ReshapeAndCacheCompressKernel);
REG_KERNEL_BASE(ReshapeAndCacheCompressRopeKernel);
REG_KERNEL_BASE(ReshapeAndCacheNzKernel);
REG_KERNEL_BASE(ReshapeAndCacheNdSisoKernel);
REG_KERNEL_BASE(ReshapeAndCacheOmniCompressKernel);
REG_KERNEL_BASE(ReshapeAndCacheF16Nd95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheBF16Nd95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheINT8Nd95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheF8e5m2Nd95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheF8e4m3Nd95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheF16CompressAlibi95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheBF16CompressAlibi95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheINT8CompressAlibi95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheF16CompressRope95Kernel);
REG_KERNEL_BASE(ReshapeAndCacheBF16CompressRope95Kernel);

} // namespace AtbOps
