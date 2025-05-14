#ifndef USR_DEFINE_PARAM_COMPARE_H
#define USR_DEFINE_PARAM_COMPARE_H
#include <mki/launch_param.h>

namespace atb {
bool IsLaunchParamEqual(const Mki::LaunchParam &launchParam1, const Mki::LaunchParam &launchParam2);
}
#endif