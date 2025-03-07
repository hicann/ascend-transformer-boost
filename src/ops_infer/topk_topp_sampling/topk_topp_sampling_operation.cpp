/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "topk_topp_sampling_operation.h"
#include "atb/utils/tensor_check.h"
#include "topk_topp_sampling_ops_runner.h"
#include "atb/utils/param_to_json.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"
#include "atb/utils/tensor_util.h"
#include "atb/core/op_param_funcs.h"

namespace atb {
static const uint32_t SINGLE_TOPK_IN_TENSOR_NUM = 2;
static const uint32_t BATCH_TOPK_MULTI_IN_TENSOR_NUM = 3;
static const uint32_t BATCH_TOPK_EXP_IN_TENSOR_NUM = 4;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t INTENSOR_TOPK = 1;
static const uint32_t INTENSOR_TOPP = 2;
static const uint32_t INTENSOR_EXP = 3;
static const uint32_t MAX_BATCH_SIZE = 512;

template <> Status CreateOperation(const infer::TopkToppSamplingParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.topkToppSamplingType <= atb::infer::TopkToppSamplingParam::SAMPLING_UNDEFINED ||
        opParam.topkToppSamplingType >= atb::infer::TopkToppSamplingParam::SAMPLING_MAX) {
        ATB_LOG(ERROR) << "topkToppSamplingType:" << opParam.topkToppSamplingType << "is invalid topkToppSamplingType";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) TopkToppSamplingOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

static Mki::OperationIr *GetOperationIrForTopkToppSampling(const infer::TopkToppSamplingParam &param)
{
    switch (param.topkToppSamplingType) {
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("TopkToppSamplingBatchTopKExpOperation");
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("TopkToppSamplingBatchTopKMultiOperation");
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::SINGLE_TOPK_SAMPLING:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("TopkToppSamplingSingleTopKOperation");
        default:
            ATB_LOG(ERROR) << "UnSupported TopkToppSamplingType: " << param.topkToppSamplingType;
    }
    return nullptr;
}

TopkToppSamplingOperation::TopkToppSamplingOperation(const infer::TopkToppSamplingParam &param)
    : OperationBase("TopkToppSamplingOperation"), param_(param)
{
    operationIr_ = GetOperationIrForTopkToppSampling(param_);
    if (!operationIr_) {
        ATB_LOG(ERROR) << "GetOperationIrForTopkToppSampling failed.";
    }
}
TopkToppSamplingOperation::~TopkToppSamplingOperation() {}

uint32_t TopkToppSamplingOperation::GetInputNum() const
{
    switch (param_.topkToppSamplingType) {
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING:
            return BATCH_TOPK_EXP_IN_TENSOR_NUM;
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING:
            return BATCH_TOPK_MULTI_IN_TENSOR_NUM;
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::SINGLE_TOPK_SAMPLING:
            return SINGLE_TOPK_IN_TENSOR_NUM;
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported TopkToppSamplingType: " << param_.topkToppSamplingType;
            return 0;
    }
}

uint32_t TopkToppSamplingOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status TopkToppSamplingOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                 SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    size_t dimNum = outTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).shape.dims[dimNum - 1] = 1;
    outTensorDescs.at(0).dtype = ACL_INT32;

    outTensorDescs.at(1) = inTensorDescs.at(0);
    outTensorDescs.at(1).shape.dims[dimNum - 1] = 1;

    return NO_ERROR;
}

Status TopkToppSamplingOperation::CheckBatchSize(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dims[0] > MAX_BATCH_SIZE) {
        ATB_LOG(ERROR) << "batch size of inTensor0: " << inTensorDescs.at(0).shape.dims[0]
                       << " is greater than max batch size " << MAX_BATCH_SIZE;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    return NO_ERROR;
}

Status TopkToppSamplingOperation::CheckSingleTopk(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.topk > inTensorDescs.at(0).shape.dims[inTensorDescs.at(0).shape.dimNum - 1]) {
        ATB_LOG(ERROR) << "topk " << param_.topk << " should be 0 or less than or equal probs dimNum("
                       << inTensorDescs.at(0).shape.dimNum << ")";
        return ERROR_INVALID_PARAM;
    }
    return CheckBatchSize(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckBatchTopkTopp(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t probsBatchSize = inTensorDescs.at(0).shape.dims[0];
    int64_t topkBatchSize = inTensorDescs.at(INTENSOR_TOPK).shape.dims[0];
    int64_t toppBatchSize = inTensorDescs.at(INTENSOR_TOPP).shape.dims[0];
    int64_t topkSize = inTensorDescs.at(INTENSOR_TOPK).shape.dims[1];
    int64_t toppSize = inTensorDescs.at(INTENSOR_TOPP).shape.dims[1];
    if (!(probsBatchSize == topkBatchSize && topkBatchSize == toppBatchSize)) {
        ATB_LOG(ERROR) << "batch size of inTensor0: " << probsBatchSize
                       << ", batch size of inTensor1: " << topkBatchSize
                       << ", batch size of inTensor2: " << toppBatchSize << " should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!(topkSize == 1 && toppSize == 1)) {
        ATB_LOG(ERROR) << "size of topk: " << topkSize << " and size of topp: " << toppSize << " should be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckBatchSize(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckMutinomial(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.randSeeds.size() != static_cast<size_t>(inTensorDescs.at(0).shape.dims[0])) {
        ATB_LOG(ERROR) << "size of randSeeds: " << param_.randSeeds.size() << " should equal to probs batch size("
                       << inTensorDescs.at(0).shape.dims[0] << ")";
        return ERROR_INVALID_PARAM;
    }
    return CheckBatchTopkTopp(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckExponential(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(0), inTensorDescs.at(INTENSOR_EXP))) {
        ATB_LOG(ERROR) << "Shape of intensor3 should be same as intensor0";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckBatchTopkTopp(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckIntensorAndParam(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(0), 2)) { // 2 = [batch, voc_size]
        ATB_LOG(ERROR) << "inTensor dim num is not support, inTensor only support 2";
        return ERROR_INVALID_TENSOR_DIM;
    }
    switch (param_.topkToppSamplingType) {
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::SINGLE_TOPK_SAMPLING:
            return CheckSingleTopk(inTensorDescs);
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING:
            return CheckMutinomial(inTensorDescs);
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING:
            return CheckExponential(inTensorDescs);
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported TopkToppSamplingType: " << param_.topkToppSamplingType;
            return ERROR_INVALID_PARAM;
    }
}

Status TopkToppSamplingOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return CheckIntensorAndParam(inTensorDescs);
}

Status TopkToppSamplingOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                 const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    SVector<TensorDesc> outTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(outTensors, outTensorDescs);
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    if (!(outTensorDescs.at(0).shape.dims[0] == outTensorDescs.at(1).shape.dims[0])) {
        ATB_LOG(ERROR) << "The batch size of outTensors should be same.";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!(outTensorDescs.at(0).shape.dims[1] == outTensorDescs.at(1).shape.dims[1])) {
        ATB_LOG(ERROR) << "The vocabulary size of outTensors should be same.";
        return ERROR_INVALID_TENSOR_DIM;
    } else {
        if (!(outTensorDescs.at(0).shape.dims[1] == 1)) {
            ATB_LOG(ERROR) << "The vocabulary size of outTensors should be 1.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return CheckIntensorAndParam(inTensorDescs);
}

std::shared_ptr<Runner> TopkToppSamplingOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<TopkToppSamplingOpsRunner>(param_);
}

nlohmann::json TopkToppSamplingOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
