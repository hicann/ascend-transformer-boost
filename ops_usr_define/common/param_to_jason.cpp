#include "param_to_jason.h"
#include "usr_define_op_params.h"

namespace atb{
template <> nlohmann::jason OpParamToJason(const infer::xxxParam &opParam)
{
    nlohmann::jason paramsJson;
    paramsJson["xxx"] = opParam.xxx;

    return paramsJson
}
} // namespace atb