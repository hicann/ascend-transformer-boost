#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/types.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include "rms_norm_lingqu_tiling.h"
#include "asdops/params/norm.h"

namespace {
constexpr size_t RMSNORM_TENSOR_IN_COUNT = 2;
constexpr size_t RMSNORM_TENSOR_OUT_COUNT = 2;
}

namespace AsdOps {
using namespace Mki;
class RmsNormLingQuKernel : public KernelBase {
public:
    explicit RmsNormLingQuKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return RmsNormLingQuTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
        "norm: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_CHECK(type == AsdOps::OpParam::Norm::RMS_NORM_LINGQU,
            "rmsnormforward: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == RMSNORM_TENSOR_IN_COUNT,
                  "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == RMSNORM_TENSOR_OUT_COUNT,
                  "output num invalid", return false);
        TensorDType dtype0 = launchParam.GetInTensor(0).desc.dtype;
        TensorDType dtype1 = launchParam.GetInTensor(1).desc.dtype;
        TensorFormat format0 = launchParam.GetInTensor(0).desc.format;
        TensorFormat format1 = launchParam.GetInTensor(1).desc.format;
        MKI_CHECK((dtype0 == TENSOR_DTYPE_FLOAT16 || dtype0 == TENSOR_DTYPE_BF16 ||
            dtype0 == TENSOR_DTYPE_FLOAT) && dtype1 == dtype0, "input dtype invalid", return false);
        MKI_CHECK(format0 == TENSOR_FORMAT_ND && format1 == format0, "input format invalid",
            return false);
        return true;
    }
};
class RmsNormLingQuF16Kernel : public RmsNormLingQuKernel {
public:
    explicit RmsNormLingQuF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : RmsNormLingQuKernel(kernelName, handle)
    {
    }
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return AsdOps::Status::OkStatus();
    }
};
REG_KERNEL_BASE(RmsNormLingQuF16Kernel);
class RmsNormLingQuBF16Kernel : public RmsNormLingQuKernel {
public:
    explicit RmsNormLingQuBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : RmsNormLingQuKernel(kernelName, handle)
    {
    }
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return AsdOps::Status::OkStatus();
    }
};
REG_KERNEL_BASE(RmsNormLingQuBF16Kernel);
} // namespace AsdOps