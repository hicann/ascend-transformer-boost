/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_parallel_lcoc_runner.h"
#include <string>
#include "atb/utils/log.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/common_utils.h"

namespace atb {
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;

LinearParallelLcocRunner::LinearParallelLcocRunner(const infer::LinearParallelParam &param)
    : LcocRunner("LinearParallelLcocRunner", RUNNER_TYPE_LINEAR_PARALLEL, param.rank, param.rankSize, param.commMode),
      param_(param)
{
    ATB_LOG(INFO) << "LinearParallelLcocRunner::LinearParallelLcocRunner called";
    switch (param_.type) {
        case infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE:
            lcalType_ = Lcal::LcalType::MATMUL_ALL_REDUCE;
            isQuant_ = param_.quantType > infer::LinearParallelParam::QuantType::QUANT_TYPE_UNDEFINED &&
                       param_.quantType < infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX;
            break;
        case infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            lcalType_ = Lcal::LcalType::MATMUL_REDUCE_SCATTER;
            break;
        case infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            lcalType_ =
                param_.keepIntermediate ? Lcal::LcalType::ALL_GATHER_MATMUL_V2 : Lcal::LcalType::ALL_GATHER_MATMUL;
            break;
        case infer::LinearParallelParam::ParallelType::PURE_LINEAR:
            lcalType_ = Lcal::LcalType::PURE_MATMUL;
            isQuant_ = param_.quantType > infer::LinearParallelParam::QuantType::QUANT_TYPE_UNDEFINED &&
                       param_.quantType < infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX;
            break;
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported type: " << param_.type;
    }
}
LinearParallelLcocRunner::~LinearParallelLcocRunner() {}

static Lcal::CoCDataTypeDesc GetCoCDataTypeDesc(const TensorDesc &input, const TensorDesc &weight,
                                                const TensorDesc &output)
{
    if (input.dtype == ACL_FLOAT16) {
        if (weight.dtype == ACL_FLOAT16) {
            return Lcal::CoCDataTypeDesc::FP16FP16_FP32_FP16;
        } else if (weight.dtype == ACL_INT8) {
            return Lcal::CoCDataTypeDesc::FP16INT8_INT32_FP16;
        }
    } else if (input.dtype == ACL_BF16) {
        if (weight.dtype == ACL_BF16) {
            return Lcal::CoCDataTypeDesc::BF16BF16_FP32_BF16;
        } else if (weight.dtype == ACL_INT8) {
            return Lcal::CoCDataTypeDesc::BF16INT8_INT32_BF16;
        }
    } else if (input.dtype == ACL_INT8 && weight.dtype == ACL_INT8) {
        return output.dtype == ACL_BF16 ? Lcal::CoCDataTypeDesc::INT8INT8_INT32_BF16 :
                                          Lcal::CoCDataTypeDesc::INT8INT8_INT32_FP16;
    }
    return Lcal::CoCDataTypeDesc::COC_DATA_TYPE_UNDEFINED;
}

static void MergeAxisInputTensorDesc(TensorDesc &input)
{
    if (input.shape.dimNum == DIM_3) {
        input.shape.dimNum = DIM_2;
        input.shape.dims[0] *= input.shape.dims[1];
        input.shape.dims[1] = input.shape.dims[DIM_2];
    }
}

Status LinearParallelLcocRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!lcoc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "lcoc_ is null, rank: " << param_.rank;
        return ERROR_INTERNAL_ERROR;
    }
    TensorDesc &input = runnerVariantPack.inTensors.at(0).desc;
    const TensorDesc &weight = runnerVariantPack.inTensors.at(1).desc;
    MergeAxisInputTensorDesc(input);
    Lcal::MatMulInfo mmInfo{.batchSize = OperationUtil::GetTensorBatch(input),
                            .m = OperationUtil::GetXTensorM(input),
                            .k = OperationUtil::GetXTensorK(input),
                            .n = OperationUtil::GetYTensorN(weight, param_.transWeight),
                            .transA = false,
                            .transB = param_.transWeight,
                            .withBias = param_.hasResidual,
                            .isInt8 = (input.dtype == ACL_INT8),
                            .weightNz = (weight.format == ACL_FORMAT_FRACTAL_NZ)};
    Lcal::CoCDataTypeDesc dataTypeDesc = GetCoCDataTypeDesc(input, weight, runnerVariantPack.outTensors.at(0).desc);
    if (dataTypeDesc == Lcal::CoCDataTypeDesc::COC_DATA_TYPE_UNDEFINED) {
        ATB_LOG(ERROR) << GetLogPrefix() << "GetCoCDataTypeDesc failed.";
        return ERROR_INTERNAL_ERROR;
    }
    Lcal::CoCParamDesc coCParamDesc{
        .dataTypeDesc = dataTypeDesc,
        .mmInfo = mmInfo,
    };
    if (isQuant_) {
        coCParamDesc.quantInfo = {static_cast<Lcal::QuantGranularity>(param_.quantType), param_.quantGroupSize};
    }
    int ret = lcoc_->SetParam(lcalType_, {}, coCParamDesc);
    if (ret != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "SetParam failed, ret : " << ret;
        return ERROR_INTERNAL_ERROR;
    }
    return NO_ERROR;
}

uint64_t LinearParallelLcocRunner::GetWorkspaceBufferSizeImpl()
{
    if (!lcoc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "lcoc_ is null, rank: " << param_.rank;
        return ERROR_INTERNAL_ERROR;
    }
    uint64_t result = static_cast<uint64_t>(lcoc_->GetWorkspaceSize());
    ATB_LOG(INFO) << GetLogPrefix() << "GetWorkspaceSize: " << result;
    return result;
};

ErrorType LinearParallelLcocRunner::LaunchKernel(Lcal::CoCInputPkg inputPkg, Lcal::CoCOutputPkg outputPkg,
                                                 RunnerVariantPack &runnerVariantPack,
                                                 infer::LinearParallelParam::ParallelType type)
{
    int ret = 0;
    switch (type) {
        case infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE:
            ret = lcoc_->MatmulAllReduce(inputPkg, outputPkg, runnerVariantPack.workspaceBuffer,
                                         runnerVariantPack.context->GetExecuteStream());
            break;
        case infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            ret = lcoc_->MatmulReduceScatter(inputPkg, outputPkg, runnerVariantPack.workspaceBuffer,
                                             runnerVariantPack.context->GetExecuteStream());
            break;
        case infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            if (param_.keepIntermediate) {
                ret = lcoc_->AllGatherMatmulV2(inputPkg, outputPkg, runnerVariantPack.workspaceBuffer,
                                               runnerVariantPack.context->GetExecuteStream());
                break;
            }
            ret = lcoc_->AllGatherMatmul(inputPkg, outputPkg, runnerVariantPack.workspaceBuffer,
                                         runnerVariantPack.context->GetExecuteStream());
            break;
        case infer::LinearParallelParam::ParallelType::PURE_LINEAR:
            ret = lcoc_->PureMatmul(inputPkg, outputPkg, runnerVariantPack.workspaceBuffer,
                                    runnerVariantPack.context->GetExecuteStream());
            break;
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported type: " << param_.type;
            return ERROR_INTERNAL_ERROR;
    }
    if (ret != Lcal::LCAL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Lcoc error: " << ret;
        return ERROR_INTERNAL_ERROR;
    }
    return NO_ERROR;
}

Status LinearParallelLcocRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!lcoc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "lcoc_ is null, rank: " << param_.rank;
        return ERROR_INTERNAL_ERROR;
    }
    size_t inTensorId = 0;
    const SVector<Tensor> &inTensors = runnerVariantPack.inTensors;
    Lcal::CoCInputPkg inputPkg;
    inputPkg.matrixA = inTensors.at(inTensorId++).deviceData;
    inputPkg.matrixB = inTensors.at(inTensorId++).deviceData;
    if (isQuant_) {
        inputPkg.dequantOffset = inTensors.at(inTensorId++).deviceData;
        inputPkg.dequantScale = inTensors.at(inTensorId++).deviceData;
    }
    inputPkg.bias = param_.hasResidual ? inTensors.at(inTensorId++).deviceData : nullptr;
    Lcal::CoCOutputPkg outputPkg = {runnerVariantPack.outTensors[0].deviceData,
                                    param_.keepIntermediate ? runnerVariantPack.outTensors[1].deviceData : nullptr};
    ErrorType ret = LaunchKernel(inputPkg, outputPkg, runnerVariantPack, param_.type);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Execute failed, ret: " << static_cast<int>(ret);
        return ERROR_INTERNAL_ERROR;
    }
    return NO_ERROR;
}
} // namespace atb