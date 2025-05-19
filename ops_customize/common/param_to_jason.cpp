#include "param_to_jason.h"
#include "customize_op_params.h"

namespace atb{
template <> nlohmann::json OpParamToJson(const customize::AddParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["param_for_nothing"] = opParam.paramForNothing;

    return paramsJson;
}
} // namespace atb