/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_operation.h"
#include "fusion_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/config.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "atb/core/op_param_funcs.h"

namespace atb {
const uint32_t TENSOR_NUM_ONE = 1;
const uint32_t TENSOR_NUM_TWO = 2;
const uint32_t TENSOR_NUM_THREE = 3;
const uint32_t TENSOR_IDX_ZERO = 0;
const uint32_t TENSOR_IDX_ONE = 1;
const uint32_t TENSOR_IDX_TWO = 2;
template <> Status CreateOperation(const infer::FusionParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);

    *operation = new (std::nothrow) FusionOperation(opParam);

    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

FusionOperation::FusionOperation(const infer::FusionParam &param) : OperationBase("FusionOperation"), param_(param)
{
    static std::map<infer::FusionParam::FusionType, std::string> opIniTable = {
        {infer::FusionParam::FusionType::MATMUL_ADD, "FusionOperationMatmulAdd"},
        {infer::FusionParam::FusionType::MATMUL_GELU, "FusionOperationMatmulGelu"},
        {infer::FusionParam::FusionType::MATMUL_SIGMOID, "FusionOperationMatmulSigmoid"},
        {infer::FusionParam::FusionType::MATMUL_SWIGLU, "FusionOperationMatmulSwiGlu"},
    };
}

FusionOperation::~FusionOperation() {}

uint32_t FusionOperation::GetInputNum() const
{
    static std::map<infer::FusionParam::FusionType, uint32_t> inTensorNumTable = {
        {infer::FusionParam::FusionType::MATMUL_ADD, TENSOR_NUM_THREE},
        {infer::FusionParam::FusionType::MATMUL_GELU, TENSOR_NUM_TWO},
        {infer::FusionParam::FusionType::MATMUL_SIGMOID, TENSOR_NUM_TWO},
        {infer::FusionParam::FusionType::MATMUL_SWIGLU, TENSOR_NUM_TWO},
    };
    std::map<infer::FusionParam::FusionType, uint32_t>::const_iterator it = inTensorNumTable.find(param_.fusionType);
    if (it != inTensorNumTable.end()) {
        return it->second;
    }
    ATB_LOG(ERROR) << "param_.fusionType is invalid, type:" << param_.fusionType;
    return NO_ERROR;
}

uint32_t FusionOperation::GetOutputNum() const
{
    return TENSOR_NUM_ONE;
}

Status FusionOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
    SVector<TensorDesc> &outTensorDescs) const
{
    if (infer::FusionParam::FusionType::NON_FUSION == param_.fusionType) {
        return NO_ERROR;
    }
    if (infer::FusionParam::FusionType::MATMUL_ADD == param_.fusionType) {
        outTensorDescs.at(TENSOR_IDX_ZERO) = inTensorDescs.at(TENSOR_IDX_TWO);
    } else {
        outTensorDescs.at(TENSOR_IDX_ZERO) = inTensorDescs.at(TENSOR_IDX_ONE);
        outTensorDescs.at(TENSOR_IDX_ZERO).shape.dims[0] = inTensorDescs.at(TENSOR_IDX_ZERO).shape.dims[0];
        outTensorDescs.at(TENSOR_IDX_ZERO).shape.dims[1] = inTensorDescs.at(TENSOR_IDX_ONE).shape.dims[0];
    }
    return NO_ERROR;
}

SVector<bool> FusionOperation::GetEmptyInTensorPermissions() const
{
    SVector<bool> v;
    if (GetInputNum() == TENSOR_NUM_THREE) {
        SVector<bool> emptyTensorPerms(GetInputNum(), false);
        emptyTensorPerms.at(TENSOR_NUM_THREE - 1) = true;
        return emptyTensorPerms;
    }
    return v;
}

std::shared_ptr<Runner> FusionOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.fusionType == infer::FusionParam::FusionType::NON_FUSION) {
        return std::make_shared<Runner>("NOT_MAIN_FUSION");
    }
    return std::make_shared<FusionOpsRunner>(param_);
}

nlohmann::json FusionOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb


