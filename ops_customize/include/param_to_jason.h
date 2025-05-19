#ifndef CUSTOMIZE_PARAM_TO_JSON_H
#define CUSTOMIZE_PARAM_TO_JSON_H
namespace atb {
template <typename OpParam> nlohmann::json OpParamToJson(const OpParam &opParam);
}
#endif