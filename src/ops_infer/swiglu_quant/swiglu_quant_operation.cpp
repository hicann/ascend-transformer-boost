/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "swiglu_quant_operation.h"
#include <limits>
#include <algorithm>
#include "swiglu_quant_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils.h"
#include "atb/utils/param_to_json.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/core/op_param_funcs.h"
 
namespace atb {
static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 2;
static const uint64_t INPUT_TENSOR_DIM_NUM = 2;
 
template <> Status CreateOperation(const infer::SwigluQuantParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ATB_LOG(INFO) << "CreateOperation SwigluQuantParam indexType: " << opParam.quantType;
    if (opParam.quantType != infer::SwigluQuantParam::QUANT_TYPE_PER_TOKEN) {
        ATB_LOG(ERROR) << "param quantType should be QUANT_TYPE_PER_TOKEN";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) SwigluQuantOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}
 
SwigluQuantOperation::SwigluQuantOperation(const infer::SwigluQuantParam &param)
    : OperationBase("SwigluQuantOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SwigluQuantOperation");
}
 
SwigluQuantOperation::~SwigluQuantOperation() {}
 
uint32_t SwigluQuantOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}
 
uint32_t SwigluQuantOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}
 
Status SwigluQuantOperation::DimCheck(const TensorDesc &inTensorDesc) const
{
    uint64_t dimNum = inTensorDesc.shape.dimNum;
    if (dimNum != INPUT_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << "dim size of inTensor should be 2, but inTensor dimNum is : " << dimNum;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}
 
Status SwigluQuantOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return DimCheck(inTensorDescs.at(0));
}
 
Status SwigluQuantOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    (void)outTensors;
    return DimCheck(inTensors.at(0).desc);
}
 
Status SwigluQuantOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).dtype = ACL_INT8;
    outTensorDescs.at(1) = inTensorDescs.at(0);
    outTensorDescs.at(1).dtype = ACL_FLOAT;
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1] / 2; // 2: chunk num
    outTensorDescs.at(1).shape.dimNum = 1;
    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    return NO_ERROR;
}
 
std::shared_ptr<Runner> SwigluQuantOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<SwigluQuantOpsRunner>(param_);
}

nlohmann::json SwigluQuantOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb