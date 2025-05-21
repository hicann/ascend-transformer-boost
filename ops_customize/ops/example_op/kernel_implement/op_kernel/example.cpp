#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "example_op/kernel_implement/tilling/example_tiling.h"

namespace AsdOps {
class AddKernel : public KernelBase {
public:
    explicit AddKernel(const std::string &kernelName, const BinHandle *handle) noexcept : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return BroadcastCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// AddF16
class AddF16Kernel : public AddKernel {
public:
    explicit AddF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AddKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AddKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 &&
                      launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT16,
                  "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "out tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(AddF16Kernel);

} // namespace AsdOps