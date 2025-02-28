/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <sstream>
#include <atb/utils/param_to_json.h>
#include "operation_wrapper.h"
#include "graph_operation_builder.h"
#include "resource/memory_manager.h"
#include "prof/prof_stats.h"

namespace py = pybind11;
using namespace atb;
using namespace atb::infer;

PYBIND11_MODULE(_C, m)
{
    m.doc() = "Python bindings for torch_atb";

    py::class_<TorchAtb::ProfStats>(m, "Prof")
        .def_static("get_prof_stats", &TorchAtb::ProfStats::GetProfStats, py::return_value_policy::reference)
        .def("get_run_time_stats", &TorchAtb::ProfStats::GetRunTimeStats);

    py::class_<TorchAtb::OperationWrapper>(m, "Operation")
        .def(py::init<const LayerNormParam &>())
        .def(py::init<const ElewiseParam &>())
        .def(py::init<const LinearParam &>())
        .def(py::init<const SoftmaxParam &>())
        .def(py::init<const SelfAttentionParam &>())
        .def(py::init<const RopeParam &>())
        .def(py::init<const SplitParam &>())
        .def(py::init<const GatherParam &>())
        .def(py::init<const ActivationParam &>())
        .def(py::init<const RmsNormParam &>())
        .def_property_readonly("name", &TorchAtb::OperationWrapper::GetName)
        .def_property_readonly("input_num", &TorchAtb::OperationWrapper::GetInputNum)
        .def_property_readonly("output_num", &TorchAtb::OperationWrapper::GetOutputNum)
        .def("forward", &TorchAtb::OperationWrapper::Forward)
        .def("__repr__", [](const TorchAtb::OperationWrapper &opWrapper) {
            std::stringstream ss;
            ss << "op name: " << opWrapper.GetName() << ", input_num: " << opWrapper.GetInputNum()
               << ", output_num: " << opWrapper.GetOutputNum();
            return ss.str();
        });

    py::class_<TorchAtb::GraphOperationBuilder>(m, "GraphBuilder")
        .def(py::init<const std::string &>())
        .def("set_input_output", &TorchAtb::GraphOperationBuilder::SetInputOutput)
        .def("reshape", &TorchAtb::GraphOperationBuilder::Reshape)
        .def("add_operation", &TorchAtb::GraphOperationBuilder::AddOperation)
        .def("build", &TorchAtb::GraphOperationBuilder::Build);

    py::enum_<aclDataType>(m, "AclDataType")
        .value("ACL_DT_UNDEFINED", aclDataType::ACL_DT_UNDEFINED)
        .value("ACL_FLOAT", aclDataType::ACL_FLOAT)
        .value("ACL_FLOAT16", aclDataType::ACL_FLOAT16)
        .value("ACL_INT8", aclDataType::ACL_INT8)
        .value("ACL_INT32", aclDataType::ACL_INT32)
        .value("ACL_UINT8", aclDataType::ACL_UINT8)
        .value("ACL_INT16", aclDataType::ACL_INT16)
        .value("ACL_UINT16", aclDataType::ACL_UINT16)
        .value("ACL_UINT32", aclDataType::ACL_UINT32)
        .value("ACL_INT64", aclDataType::ACL_INT64)
        .value("ACL_UINT64", aclDataType::ACL_UINT64)
        .value("ACL_DOUBLE", aclDataType::ACL_DOUBLE)
        .value("ACL_BOOL", aclDataType::ACL_BOOL)
        .value("ACL_STRING", aclDataType::ACL_STRING)
        .value("ACL_COMPLEX64", aclDataType::ACL_COMPLEX64)
        .value("ACL_COMPLEX128", aclDataType::ACL_COMPLEX128)
        .value("ACL_BF16", aclDataType::ACL_BF16)
        .value("ACL_INT4", aclDataType::ACL_INT4)
        .value("ACL_UINT1", aclDataType::ACL_UINT1)
        .value("ACL_COMPLEX32", aclDataType::ACL_COMPLEX32)
        .export_values();

    py::enum_<aclFormat>(m, "AclFormat")
        .value("ACL_FORMAT_UNDEFINED", ACL_FORMAT_UNDEFINED)
        .value("ACL_FORMAT_NCHW", ACL_FORMAT_NCHW)
        .value("ACL_FORMAT_NHWC", ACL_FORMAT_NHWC)
        .value("ACL_FORMAT_ND", ACL_FORMAT_ND)
        .value("ACL_FORMAT_NC1HWC0", ACL_FORMAT_NC1HWC0)
        .value("ACL_FORMAT_FRACTAL_Z", ACL_FORMAT_FRACTAL_Z)
        .value("ACL_FORMAT_NC1HWC0_C04", ACL_FORMAT_NC1HWC0_C04)
        .value("ACL_FORMAT_HWCN", ACL_FORMAT_HWCN)
        .value("ACL_FORMAT_NDHWC", ACL_FORMAT_NDHWC)
        .value("ACL_FORMAT_FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ)
        .value("ACL_FORMAT_NCDHW", ACL_FORMAT_NCDHW)
        .value("ACL_FORMAT_NDC1HWC0", ACL_FORMAT_NDC1HWC0)
        .value("ACL_FRACTAL_Z_3D", ACL_FRACTAL_Z_3D)
        .export_values();

    py::enum_<InputLayout>(m, "InputLayout")
        .value("TYPE_BSND", InputLayout::TYPE_BSND)
        .value("TYPE_BNSD", InputLayout::TYPE_BNSD)
        .export_values();

    py::enum_<QuantType>(m, "QuantType")
        .value("QUANT_UNDEFINED", QuantType::QUANT_UNDEFINED)
        .value("QUANT_INT4", QuantType::QUANT_INT4)
        .value("QUANT_INT8", QuantType::QUANT_INT8)
        .value("QUANT_INT16", QuantType::QUANT_INT16)
        .value("QUANT_FLOAT8", QuantType::QUANT_FLOAT8)
        .value("QUANT_FLOAT16", QuantType::QUANT_FLOAT16)
        .export_values();

    py::enum_<DynamicQuantType>(m, "DynamicQuantType")
        .value("DYNAMIC_QUANT_UNDEFINED", DynamicQuantType::DYNAMIC_QUANT_UNDEFINED)
        .value("DYNAMIC_QUANT_SYMMETRIC", DynamicQuantType::DYNAMIC_QUANT_SYMMETRIC)
        .value("DYNAMIC_QUANT_ASYMMETRIC", DynamicQuantType::DYNAMIC_QUANT_ASYMMETRIC)
        .export_values();

    py::class_<LayerNormParam> layerNorm(m, "LayerNormParam");
    layerNorm.def(py::init<>())
        .def_readwrite("layer_type", &LayerNormParam::layerType)
        .def_readwrite("norm_param", &LayerNormParam::normParam)
        .def_readwrite("pre_norm_param", &LayerNormParam::preNormParam)
        .def_readwrite("post_norm_param", &LayerNormParam::postNormParam)
        .def("__repr__", [](const LayerNormParam &param) { return "LayerNormParam: " + OpParamToJson(param).dump(); });

    py::enum_<LayerNormParam::LayerNormType>(layerNorm, "LayerNormType")
        .value("LAYER_NORM_UNDEFINED", LayerNormParam::LayerNormType::LAYER_NORM_UNDEFINED)
        .value("LAYER_NORM_NORM", LayerNormParam::LayerNormType::LAYER_NORM_NORM)
        .value("LAYER_NORM_PRENORM", LayerNormParam::LayerNormType::LAYER_NORM_PRENORM)
        .value("LAYER_NORM_POSTNORM", LayerNormParam::LayerNormType::LAYER_NORM_POSTNORM)
        .value("LAYER_NORM_MAX", LayerNormParam::LayerNormType::LAYER_NORM_MAX);

    py::class_<LayerNormParam::NormParam>(layerNorm, "NormParam")
        .def(py::init<>())
        .def_readwrite("quant_type", &LayerNormParam::NormParam::quantType)
        .def_readwrite("epsilon", &LayerNormParam::NormParam::epsilon)
        .def_readwrite("begin_norm_axis", &LayerNormParam::NormParam::beginNormAxis)
        .def_readwrite("begin_params_axis", &LayerNormParam::NormParam::beginParamsAxis)
        .def_readwrite("dynamic_quant_type", &LayerNormParam::NormParam::dynamicQuantType);

    py::class_<LayerNormParam::PreNormParam>(layerNorm, "PreNormParam")
        .def(py::init<>())
        .def_readwrite("quant_type", &LayerNormParam::PreNormParam::quantType)
        .def_readwrite("epsilon", &LayerNormParam::PreNormParam::epsilon)
        .def_readwrite("op_mode", &LayerNormParam::PreNormParam::opMode)
        .def_readwrite("zoom_scale_value", &LayerNormParam::PreNormParam::zoomScaleValue);

    py::class_<LayerNormParam::PostNormParam>(layerNorm, "PostNormParam")
        .def(py::init<>())
        .def_readwrite("quant_type", &LayerNormParam::PostNormParam::quantType)
        .def_readwrite("epsilon", &LayerNormParam::PostNormParam::epsilon)
        .def_readwrite("op_mode", &LayerNormParam::PostNormParam::opMode)
        .def_readwrite("zoom_scale_value", &LayerNormParam::PostNormParam::zoomScaleValue);

    py::class_<ElewiseParam> elewise(m, "ElewiseParam");
    elewise.def(py::init<>())
        .def_readwrite("elewise_type", &ElewiseParam::elewiseType)
        .def_readwrite("quant_param", &ElewiseParam::quantParam)
        .def_readwrite("muls_param", &ElewiseParam::mulsParam)
        .def_readwrite("out_tensor_type", &ElewiseParam::outTensorType)
        .def("__repr__", [](const ElewiseParam &param) { return "ElewiseParam: " + OpParamToJson(param).dump(); });

    py::enum_<ElewiseParam::ElewiseType>(elewise, "ElewiseType")
        .value("ELEWISE_UNDEFINED", ElewiseParam::ElewiseType::ELEWISE_UNDEFINED)
        .value("ELEWISE_CAST", ElewiseParam::ElewiseType::ELEWISE_CAST)
        .value("ELEWISE_MULS", ElewiseParam::ElewiseType::ELEWISE_MULS)
        .value("ELEWISE_COS", ElewiseParam::ElewiseType::ELEWISE_COS)
        .value("ELEWISE_SIN", ElewiseParam::ElewiseType::ELEWISE_SIN)
        .value("ELEWISE_NEG", ElewiseParam::ElewiseType::ELEWISE_NEG)
        .value("ELEWISE_QUANT", ElewiseParam::ElewiseType::ELEWISE_QUANT)
        .value("ELEWISE_LOGICAL_NOT", ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT)
        .value("ELEWISE_ADD", ElewiseParam::ElewiseType::ELEWISE_ADD)
        .value("ELEWISE_MUL", ElewiseParam::ElewiseType::ELEWISE_MUL)
        .value("ELEWISE_REALDIV", ElewiseParam::ElewiseType::ELEWISE_REALDIV)
        .value("ELEWISE_LOGICAL_AND", ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND)
        .value("ELEWISE_LOGICAL_OR", ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR)
        .value("ELEWISE_LESS", ElewiseParam::ElewiseType::ELEWISE_LESS)
        .value("ELEWISE_GREATER", ElewiseParam::ElewiseType::ELEWISE_GREATER)
        .value("ELEWISE_SUB", ElewiseParam::ElewiseType::ELEWISE_SUB)
        .value("ELEWISE_EQUAL", ElewiseParam::ElewiseType::ELEWISE_EQUAL)
        .value("ELEWISE_QUANT_PER_CHANNEL", ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL)
        .value("ELEWISE_DEQUANT_PER_CHANNEL", ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL)
        .value("ELEWISE_DYNAMIC_QUANT", ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT)
        .value("ELEWISE_TANH", ElewiseParam::ElewiseType::ELEWISE_TANH);

    py::class_<ElewiseParam::QuantParam>(elewise, "QuantParam")
        .def(py::init<>())
        .def_readwrite("input_scale", &ElewiseParam::QuantParam::inputScale)
        .def_readwrite("asymmetric", &ElewiseParam::QuantParam::asymmetric)
        .def_readwrite("input_offset", &ElewiseParam::QuantParam::inputOffset);

    py::class_<ElewiseParam::MulsParam>(elewise, "MulsParam")
        .def(py::init<>())
        .def_readwrite("var_attr", &ElewiseParam::MulsParam::varAttr);

    py::class_<LinearParam>(m, "LinearParam")
        .def(py::init<>())
        .def_readwrite("transpose_a", &LinearParam::transposeA)
        .def_readwrite("transpose_b", &LinearParam::transposeB)
        .def_readwrite("has_bias", &LinearParam::hasBias)
        .def_readwrite("out_data_type", &LinearParam::outDataType)
        .def_readwrite("en_accum", &LinearParam::enAccum)
        .def("__repr__", [](const LinearParam &param) { return "LinearParam: " + OpParamToJson(param).dump(); });

    py::class_<SoftmaxParam>(m, "SoftmaxParam")
        .def(py::init<>())
        .def_readwrite("axes", &SoftmaxParam::axes)
        .def_property(
            "axes",
            [](SoftmaxParam &param) -> std::vector<int64_t> {
                std::vector<int64_t> axisVec;
                axisVec.resize(param.axes.size());
                for (size_t i = 0; i < param.axes.size(); i++) {
                    axisVec.at(i) = param.axes.at(i);
                }
                return axisVec;
            },
            [](SoftmaxParam &param, const std::vector<int64_t> &axes) {
                param.axes.clear();
                for (const auto &axis : axes) {
                    param.axes.push_back(axis);
                }
            })
        .def("__repr__", [](const SoftmaxParam &param) { return "SoftmaxParam: " + OpParamToJson(param).dump(); });

    py::class_<SelfAttentionParam> selfAttention(m, "SelfAttentionParam");
    selfAttention.def(py::init<>())
        .def_readwrite("quant_type", &SelfAttentionParam::quantType)
        .def_readwrite("out_data_type", &SelfAttentionParam::outDataType)
        .def_readwrite("head_num", &SelfAttentionParam::headNum)
        .def_readwrite("kv_head_num", &SelfAttentionParam::kvHeadNum)
        .def_readwrite("q_scale", &SelfAttentionParam::qScale)
        .def_readwrite("qk_scale", &SelfAttentionParam::qkScale)
        .def_readwrite("batch_run_status_enable", &SelfAttentionParam::batchRunStatusEnable)
        .def_readwrite("is_triu_mask", &SelfAttentionParam::isTriuMask)
        .def_readwrite("calc_type", &SelfAttentionParam::calcType)
        .def_readwrite("kernel_type", &SelfAttentionParam::kernelType)
        .def_readwrite("clamp_type", &SelfAttentionParam::clampType)
        .def_readwrite("clamp_min", &SelfAttentionParam::clampMin)
        .def_readwrite("clamp_max", &SelfAttentionParam::clampMax)
        .def_readwrite("mask_type", &SelfAttentionParam::maskType)
        .def_readwrite("kvcache_cfg", &SelfAttentionParam::kvcacheCfg)
        .def_readwrite("scale_type", &SelfAttentionParam::scaleType)
        .def_readwrite("input_layout", &SelfAttentionParam::inputLayout)
        .def_readwrite("mla_v_head_size", &SelfAttentionParam::mlaVHeadSize)
        .def_readwrite("cache_type", &SelfAttentionParam::cacheType)
        .def_readwrite("window_size", &SelfAttentionParam::windowSize)
        .def("__repr__",
             [](const SelfAttentionParam &param) { return "SelfAttentionParam: " + OpParamToJson(param).dump(); });

    py::enum_<SelfAttentionParam::CalcType>(selfAttention, "CalcType")
        .value("UNDEFINED", SelfAttentionParam::UNDEFINED)
        .value("ENCODER", SelfAttentionParam::ENCODER)
        .value("DECODER", SelfAttentionParam::DECODER)
        .value("PA_ENCODER", SelfAttentionParam::PA_ENCODER);

    py::enum_<SelfAttentionParam::KernelType>(selfAttention, "KernelType")
        .value("KERNELTYPE_DEFAULT", SelfAttentionParam::KERNELTYPE_DEFAULT)
        .value("KERNELTYPE_HIGH_PRECISION", SelfAttentionParam::KERNELTYPE_HIGH_PRECISION);

    py::enum_<SelfAttentionParam::ClampType>(selfAttention, "ClampType")
        .value("CLAMP_TYPE_UNDEFINED", SelfAttentionParam::CLAMP_TYPE_UNDEFINED)
        .value("CLAMP_TYPE_MIN_MAX", SelfAttentionParam::CLAMP_TYPE_MIN_MAX);

    py::enum_<SelfAttentionParam::MaskType>(selfAttention, "MaskType")
        .value("MASK_TYPE_UNDEFINED", SelfAttentionParam::MASK_TYPE_UNDEFINED)
        .value("MASK_TYPE_NORM", SelfAttentionParam::MASK_TYPE_NORM)
        .value("MASK_TYPE_ALIBI", SelfAttentionParam::MASK_TYPE_ALIBI)
        .value("MASK_TYPE_NORM_COMPRESS", SelfAttentionParam::MASK_TYPE_NORM_COMPRESS)
        .value("MASK_TYPE_ALIBI_COMPRESS", SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS)
        .value("MASK_TYPE_ALIBI_COMPRESS_SQRT", SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT)
        .value("MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN", SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN)
        .value("MASK_TYPE_SLIDING_WINDOW_NORM", SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM)
        .value("MASK_TYPE_SLIDING_WINDOW_COMPRESS", SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS);

    py::enum_<SelfAttentionParam::KvCacheCfg>(selfAttention, "KvCacheCfg")
        .value("K_CACHE_V_CACHE", SelfAttentionParam::K_CACHE_V_CACHE)
        .value("K_BYPASS_V_BYPASS", SelfAttentionParam::K_BYPASS_V_BYPASS);

    py::enum_<SelfAttentionParam::ScaleType>(selfAttention, "ScaleType")
        .value("SCALE_TYPE_TOR", SelfAttentionParam::SCALE_TYPE_TOR)
        .value("SCALE_TYPE_LOGN", SelfAttentionParam::SCALE_TYPE_LOGN)
        .value("SCALE_TYPE_MAX", SelfAttentionParam::SCALE_TYPE_MAX);

    py::enum_<SelfAttentionParam::QuantType>(selfAttention, "QuantType")
        .value("TYPE_QUANT_UNDEFINED", SelfAttentionParam::TYPE_QUANT_UNDEFINED)
        .value("TYPE_DEQUANT_FUSION", SelfAttentionParam::TYPE_DEQUANT_FUSION)
        .value("TYPE_QUANT_QKV_OFFLINE", SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE)
        .value("TYPE_QUANT_QKV_ONLINE", SelfAttentionParam::TYPE_QUANT_QKV_ONLINE);
    
    py::enum_<SelfAttentionParam::CacheType>(selfAttention, "CacheType")
        .value("CACHE_TYPE_NORM", SelfAttentionParam::CACHE_TYPE_NORM)
        .value("CACHE_TYPE_SWA", SelfAttentionParam::CACHE_TYPE_SWA);

    py::class_<RopeParam>(m, "RopeParam")
        .def(py::init<>())
        .def_readwrite("rotary_coeff", &RopeParam::rotaryCoeff)
        .def_readwrite("cos_format", &RopeParam::cosFormat)
        .def("__repr__", [](const RopeParam &param) { return "RopeParam: " + OpParamToJson(param).dump(); });

    py::class_<SplitParam>(m, "SplitParam")
        .def(py::init<>())
        .def_readwrite("split_dim", &SplitParam::splitDim)
        .def_readwrite("split_num", &SplitParam::splitNum)
        .def_property(
            "split_sizes",
            [](SplitParam &param) -> std::vector<int32_t> {
                std::vector<int32_t> vec;
                vec.resize(param.splitSizes.size());
                for (size_t i = 0; i < param.splitSizes.size(); i++) {
                    vec.at(i) = param.splitSizes.at(i);
                }
                return vec;
            },
            [](SplitParam &param, const std::vector<int32_t> &splitSizes) {
                param.splitSizes.clear();
                for (const auto &splitSize : splitSizes) {
                    param.splitSizes.push_back(splitSize);
                }
            })
        .def("__repr__", [](const SplitParam &param) { return "SplitParam: " + OpParamToJson(param).dump(); });

    py::class_<GatherParam>(m, "GatherParam")
        .def(py::init<>())
        .def_readwrite("axis", &GatherParam::axis)
        .def_readwrite("batch_dims", &GatherParam::batchDims)
        .def("__repr__", [](const GatherParam &param) { return "GatherParam: " + OpParamToJson(param).dump(); });

    py::enum_<ActivationType>(m, "ActivationType")
        .value("ACTIVATION_UNDEFINED", ACTIVATION_UNDEFINED)
        .value("ACTIVATION_RELU", ACTIVATION_RELU)
        .value("ACTIVATION_GELU", ACTIVATION_GELU)
        .value("ACTIVATION_FAST_GELU", ACTIVATION_FAST_GELU)
        .value("ACTIVATION_SWISH", ACTIVATION_SWISH)
        .value("ACTIVATION_LOG", ACTIVATION_LOG)
        .value("ACTIVATION_SWIGLU_FORWARD", ACTIVATION_SWIGLU_FORWARD)
        .value("ACTIVATION_SWIGLU_BACKWARD", ACTIVATION_SWIGLU_BACKWARD)
        .value("ACTIVATION_SIGMOID", ACTIVATION_SIGMOID)
        .value("ACTIVATION_FASTER_GELU_FORWARD", ACTIVATION_FASTER_GELU_FORWARD)
        .value("ACTIVATION_MAX", ACTIVATION_MAX)
        .export_values();

    py::class_<ActivationParam> activation(m, "ActivationParam");
    activation.def(py::init<>())
        .def_readwrite("activation_type", &ActivationParam::activationType)
        .def_readwrite("scale", &ActivationParam::scale)
        .def_readwrite("dim", &ActivationParam::dim)
        .def_readwrite("gelu_mode", &ActivationParam::geluMode)
        .def("__repr__",
             [](const ActivationParam &param) { return "ActivationParam: " + OpParamToJson(param).dump(); });
             
    py::enum_<ActivationParam::GeLUMode>(activation, "GeLUMode")
        .value("TANH_MODE", ActivationParam::TANH_MODE)
        .value("NONE_MODE", ActivationParam::NONE_MODE);

    py::class_<RmsNormParam> rmsNorm(m, "RmsNormParam");
    rmsNorm.def(py::init<>())
        .def_readwrite("layer_type", &RmsNormParam::layerType)
        .def_readwrite("norm_param", &RmsNormParam::normParam)
        .def_readwrite("pre_norm_param", &RmsNormParam::preNormParam)
        .def_readwrite("post_norm_param", &RmsNormParam::postNormParam)
        .def("__repr__", [](const RmsNormParam &param) { return "RmsNormParam: " + OpParamToJson(param).dump(); });

    py::enum_<RmsNormParam::RmsNormType>(rmsNorm, "RmsNormType")
        .value("RMS_NORM_UNDEFINED", RmsNormParam::RMS_NORM_UNDEFINED)
        .value("RMS_NORM_NORM", RmsNormParam::RMS_NORM_NORM)
        .value("RMS_NORM_PRENORM", RmsNormParam::RMS_NORM_PRENORM)
        .value("RMS_NORM_POSTNORM", RmsNormParam::RMS_NORM_POSTNORM);

    py::enum_<RmsNormParam::PrecisionMode>(rmsNorm, "PrecisionMode")
        .value("HIGH_PRECISION_MODE", RmsNormParam::HIGH_PRECISION_MODE)
        .value("HIGH_PERFORMANCE_MODE", RmsNormParam::HIGH_PERFORMANCE_MODE);

    py::enum_<RmsNormParam::ModelType>(rmsNorm, "ModelType")
        .value("LLAMA_MODEL", RmsNormParam::LLAMA_MODEL)
        .value("GEMMA_MODEL", RmsNormParam::GEMMA_MODEL);

    py::class_<RmsNormParam::NormParam>(rmsNorm, "NormParam")
        .def(py::init<>())
        .def_readwrite("quant_type", &RmsNormParam::NormParam::quantType)
        .def_readwrite("epsilon", &RmsNormParam::NormParam::epsilon)
        .def_readwrite("layer_norm_eps", &RmsNormParam::NormParam::layerNormEps)
        .def_readwrite("rstd", &RmsNormParam::NormParam::rstd)
        .def_readwrite("precision_mode", &RmsNormParam::NormParam::precisionMode)
        .def_readwrite("model_type", &RmsNormParam::NormParam::modelType)
        .def_readwrite("dynamic_quant_type", &RmsNormParam::NormParam::dynamicQuantType);

    py::class_<RmsNormParam::PreNormParam>(rmsNorm, "PreNormParam")
        .def(py::init<>())
        .def_readwrite("quant_type", &RmsNormParam::PreNormParam::quantType)
        .def_readwrite("epsilon", &RmsNormParam::PreNormParam::epsilon)
        .def_readwrite("has_bias", &RmsNormParam::PreNormParam::hasBias);

    py::class_<RmsNormParam::PostNormParam>(rmsNorm, "PostNormParam")
        .def(py::init<>())
        .def_readwrite("quant_type", &RmsNormParam::PostNormParam::quantType)
        .def_readwrite("epsilon", &RmsNormParam::PostNormParam::epsilon)
        .def_readwrite("has_bias", &RmsNormParam::PostNormParam::hasBias);
}
