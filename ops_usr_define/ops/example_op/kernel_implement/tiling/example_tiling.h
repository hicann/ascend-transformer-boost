#ifndef CUSTOMIZE_ADD_TILING_H
#define CUSTOMIZE_ADD_TILING_H

#include <mki/bin_handle.h>
#include <mki/launch_param.h>
#include <mki/kernel_info.h>

namespace AsdOps {
using namespace Mki;
Status BroadcastCommonTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const BinHandle &binHandle);
} // namespace AsdOps
#endif