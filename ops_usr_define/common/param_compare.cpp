#include "param_compare.h"
#include <functional>
#include <map>
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include <mki/launch_param.h>
#include "atb/utils/log.h"
#include "customize_op_params.h"
#include "atb/utils/tensor_util.h"

namespace atb {
using ParamCompareFunc = std::function<bool(const Mki::Any &, const Mki::Any &)>;

template <typename T> bool ParamCompareFuncImpl(const Mki::Any &any1, const Mki::Any &any2)
{
    const auto &content1 = Mki::AnyCast<T>(any1);
    const auto &content2 = Mki::AnyCast<T>(any2);
    return content1 == content2;
}

static const std::map<std::size_t, ParamCompareFunc> USR_PARAM_COMPARE_MAP = {
    {typeid(AsdOps::OpParam::Elewise).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Elewise>},
};

bool IsLaunchParamEqual(const Mki::LaunchParam &launchParam1, const Mki::LaunchParam &launchParam2)
{
    if (launchParam1.GetInTensorCount() != launchParam2.GetInTensorCount()) {
        return false;
    }

    for (size_t i = 0; i < launchParam1.GetInTensorCount(); ++i) {
        if (!TensorUtil::AsdOpsTensorDescEqual(launchParam1.GetInTensor(i).desc, launchParam2.GetInTensor(i).desc)) {
            return false;
        }
    }

    const Mki::Any &specificParam1 = launchParam1.GetParam();
    const Mki::Any &specificParam2 = launchParam2.GetParam();
    auto it = USR_PARAM_COMPARE_MAP.find(specificParam1.Type().hash_code());
    if (it != USR_PARAM_COMPARE_MAP.end()) {
        return it->second(specificParam1, specificParam2);
    } else {
        ATB_LOG(WARN) << "Can not compare param of " << specificParam1.Type().name();
        return false;
    }
}
} // namespace atb