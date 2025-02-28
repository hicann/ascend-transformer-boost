/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multinomial_operation.h"
#include "atb/utils/log.h"
#include "multinomial_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/core/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint64_t INPUT_TENSOR_DIM_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint64_t OUT_TENSOR_DIM_NUM = 2;
static const uint64_t MAX_NUMSAMPLES = 64;

template <> Status CreateOperation(const infer::MultinomialParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new MultinomialOperation(opParam);
    return NO_ERROR;
}

MultinomialOperation::MultinomialOperation(const infer::MultinomialParam &param)
    : OperationBase("MultinomialOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("MultinomialOperation");
}

MultinomialOperation::~MultinomialOperation() {}

uint32_t MultinomialOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t MultinomialOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status MultinomialOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = param_.numSamples;
    outTensorDescs.at(0).shape.dimNum = 2; // dim: 2
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = ACL_INT32;
    return NO_ERROR;
}

Status MultinomialOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = DimCheck(inTensorDescs.at(0));
    if (st != NO_ERROR) {
        return st;
    }

    return ParamCheck(inTensorDescs.at(0));
}

Status MultinomialOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (outTensors.at(0).desc.shape.dimNum != OUT_TENSOR_DIM_NUM ||
        outTensors.at(0).desc.shape.dims[1] != param_.numSamples) {
        ATB_LOG(ERROR) << "outTensors dims is invalid, dims[1] should be param_.numSamples";
        return ERROR_INVALID_TENSOR_DIM;
    }

    Status st = DimCheck(inTensors.at(0).desc);
    if (st != NO_ERROR) {
        return st;
    }

    return ParamCheck(inTensors.at(0).desc);
}

std::shared_ptr<Runner> MultinomialOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<MultinomialOpsRunner>(param_);
}

Status MultinomialOperation::ParamCheck(const TensorDesc &inTensorDesc) const
{
    uint64_t dimNum = inTensorDesc.shape.dimNum;
    int64_t lastDim = inTensorDesc.shape.dims[dimNum - 1];
    if (param_.numSamples > static_cast<uint64_t>(lastDim) || param_.numSamples > MAX_NUMSAMPLES) {
        ATB_LOG(ERROR) << "numSamples shoud not bigger than last dim and 64, numSamples: " << param_.numSamples
                       << ", last dim: " << lastDim;
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status MultinomialOperation::DimCheck(const TensorDesc &inTensorDesc) const
{
    uint64_t dimNum = inTensorDesc.shape.dimNum;
    if (dimNum != INPUT_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << "dim size of inTensor should be 2, but inTensor dimNum is : " << dimNum;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

nlohmann::json MultinomialOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb