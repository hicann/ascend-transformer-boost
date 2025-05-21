/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_parallel_operation.h"
#include <sstream>
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "linear_parallel_graph_runner.h"
#include "linear_parallel_lcoc_runner.h"
#include "atb/utils/singleton.h"
#include "atb/core/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM_WITH_RESIDUAL = 3;
static const uint32_t IN_TENSOR_NUM_WITHOUT_RESIDUAL = 2;
static const uint32_t EXTRA_IN_TENSOR_NUM_WITH_QUANT = 2;
static const uint32_t EXTRA_IN_TENSOR_NUM_WITH_PER_TOKEN_QUANT = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM_WITH_MID = 2;

template <> Status CreateOperation(const infer::LinearParallelParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (OperationUtil::DistributedInitCheck<infer::LinearParallelParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "LinearParallelOperation DistributedInitCheck failed.";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend != "hccl" && opParam.backend != "lccl" && opParam.backend != "lcoc") {
        ATB_LOG(ERROR) << "LinearParallel backend support hccl/lccl/lcoc but get [" << opParam.backend << "]";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lcoc") {
        if (opParam.type <= atb::infer::LinearParallelParam::ParallelType::UNDEFINED ||
            opParam.type >= atb::infer::LinearParallelParam::ParallelType::MAX) {
            ATB_LOG(ERROR) << "LinearParallel type:" << opParam.type << " is invalid ParallelType";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.type != atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR &&
            opParam.keepIntermediate) {
            ATB_LOG(ERROR) << "LinearParallel backend lcoc only ALL_GATHER_LINEAR support keepIntermediate is true";
            return ERROR_INVALID_PARAM;
        }
    }
    *operation = new (std::nothrow) LinearParallelOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

LinearParallelOperation::LinearParallelOperation(const infer::LinearParallelParam &param)
    : OperationBase("LinearParallelOperation"), param_(param)
{
    commonCheckParam_ = param_;
    std::stringstream opIrKeySs;
    std::string withStr = param_.hasResidual ? "With" : "Without";
    opIrKeySs << "LinearParallelOperation" << withStr << "Residual";
    if (param_.backend == "lcoc") {
        opIrKeySs << "Lcoc";
        if (param_.keepIntermediate) {
            opIrKeySs << "KeepIntermediate";
        }
        if (commonCheckParam_.isQuant) {
            opIrKeySs << "Quant";
            if (param_.quantType == atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN) {
                opIrKeySs << "PerToken";
            }
        }
    }
    std::string opIrKey = opIrKeySs.str();
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKey);
}

LinearParallelOperation::~LinearParallelOperation() {}

uint32_t LinearParallelOperation::GetInputNum() const
{
    uint32_t inTensorNum = param_.hasResidual ? IN_TENSOR_NUM_WITH_RESIDUAL : IN_TENSOR_NUM_WITHOUT_RESIDUAL;
    if (commonCheckParam_.isQuant) {
        inTensorNum += EXTRA_IN_TENSOR_NUM_WITH_QUANT;
        if (param_.quantType == atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN) {
            inTensorNum += EXTRA_IN_TENSOR_NUM_WITH_PER_TOKEN_QUANT;
        }
    }
    return inTensorNum;
}

uint32_t LinearParallelOperation::GetOutputNum() const
{
    return param_.keepIntermediate ? OUT_TENSOR_NUM_WITH_MID : OUT_TENSOR_NUM;
}

Status LinearParallelOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                               SVector<TensorDesc> &outTensorDescs) const
{
    if (param_.backend != "lcoc") {
        return OperationUtil::MatmulInferShape(inTensorDescs, outTensorDescs, commonCheckParam_);
    }
    switch (param_.type) {
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE:
        case atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR:
            return OperationUtil::MatmulInferShape(inTensorDescs, outTensorDescs, commonCheckParam_);
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            return InferShapeLinearReduceScatter(inTensorDescs, outTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            return InferShapeAllGatherLinear(inTensorDescs, outTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR_REDUCE_SCATTER:
            return InferShapeAllGatherLinearReduceScatter(inTensorDescs, outTensorDescs);
        default:
            ATB_LOG(ERROR) << "not support type:" << param_.type;
            return ERROR_INVALID_PARAM;
    }
}


Status LinearParallelOperation::InferShapeLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs,
                                                              SVector<TensorDesc> &outTensorDescs) const
{
    Status st = OperationUtil::MatmulInferShape(inTensorDescs, outTensorDescs, commonCheckParam_);
    if (st != NO_ERROR) {
        return st;
    }
    outTensorDescs.at(0).shape.dims[0] /= param_.rankSize;
    return NO_ERROR;
}

Status LinearParallelOperation::InferShapeAllGatherLinear(const SVector<TensorDesc> &inTensorDescs,
                                                          SVector<TensorDesc> &outTensorDescs) const
{
    SVector<TensorDesc> newInTensorDescs;
    newInTensorDescs = inTensorDescs;
    newInTensorDescs.at(0).shape.dims[0] *= param_.rankSize;
    Status st = OperationUtil::MatmulInferShape(newInTensorDescs, outTensorDescs, commonCheckParam_);
    if (st != NO_ERROR) {
        return st;
    }
    if (outTensorDescs.size() == OUT_TENSOR_NUM_WITH_MID) {
        outTensorDescs.at(1) = inTensorDescs.at(0);
        outTensorDescs.at(1).shape.dims[0] *= param_.rankSize;
    }
    return NO_ERROR;
}

Status LinearParallelOperation::InferShapeAllGatherLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs,
                                                                       SVector<TensorDesc> &outTensorDescs) const
{
    SVector<TensorDesc> newInTensorDescs;
    newInTensorDescs = inTensorDescs;
    newInTensorDescs.at(0).shape.dims[0] *= param_.twoDimTPInfo.agDim;
    Status st = OperationUtil::MatmulInferShape(newInTensorDescs, outTensorDescs, commonCheckParam_);
    if (st != NO_ERROR) {
        return st;
    }
    outTensorDescs.at(0).shape.dims[0] /= param_.twoDimTPInfo.rsDim;
    return NO_ERROR;
}

Status LinearParallelOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.backend != "lcoc") {
        return InferShapeCheckLinearAllReduce(inTensorDescs);
    }
    switch (param_.type) {
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE:
        case atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR:
            return InferShapeCheckLinearAllReduce(inTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            return InferShapeCheckLinearReduceScatter(inTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            return InferShapeCheckAllGatherLinear(inTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR_REDUCE_SCATTER:
            return InferShapeCheckAllGatherLinearReduceScatter(inTensorDescs);
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "not support type:" << param_.type;
            return ERROR_INVALID_PARAM;
    }
}

Status LinearParallelOperation::CheckResidual(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.hasResidual) {
        int64_t n = OperationUtil::GetYTensorN(inTensorDescs.at(1), param_.transWeight);
        size_t residualTensorId = inTensorDescs.size() - 1;
        const TensorDesc &residual = inTensorDescs.at(residualTensorId);
        int64_t needFirstDim = 1;
        if (!OperationUtil::LinearBiasDeqCheck(residual, GetLogPrefix(), n, needFirstDim, residualTensorId)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Check residual tensor failed.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status LinearParallelOperation::InferShapeCheckLinearAllReduce(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!OperationUtil::MatmulInTensorDescsCheck(inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }

    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::InferShapeCheckLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!OperationUtil::MatmulInTensorDescsCheck(inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t xTensorM = OperationUtil::GetXTensorM(inTensorDescs.at(0));
    if (xTensorM % param_.rankSize != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 m [" << xTensorM
                       << "] should be an integer multiple of rankSize :" << param_.rankSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::InferShapeCheckAllGatherLinear(const SVector<TensorDesc> &inTensorDescs) const
{
    SVector<TensorDesc> newInTensorDescs;
    newInTensorDescs = inTensorDescs;
    newInTensorDescs.at(0).shape.dims[0] *= param_.rankSize;
    ATB_LOG(INFO) << GetLogPrefix() << "matmul input dim0: " << newInTensorDescs.at(0).shape.dims[0];
    if (!OperationUtil::MatmulInTensorDescsCheck(newInTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::InferShapeCheckAllGatherLinearReduceScatter(
    const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.twoDimTPInfo.rsDim * param_.twoDimTPInfo.agDim != param_.rankSize) {
        ATB_LOG(ERROR) << "agDim * rsDim should equal to rankSize";
        return ERROR_INVALID_PARAM;
    }
    if (param_.twoDimTPInfo.rsDim == 0) {
        ATB_LOG(ERROR) << "rsDim can't be 0";
        return ERROR_INVALID_PARAM;
    }
    SVector<TensorDesc> newInTensorDescs;
    newInTensorDescs = inTensorDescs;
    newInTensorDescs.at(0).shape.dims[0] *= param_.twoDimTPInfo.agDim;
    ATB_LOG(INFO) << GetLogPrefix() << "matmul input dim0: " << newInTensorDescs.at(0).shape.dims[0];
    if (!OperationUtil::MatmulInTensorDescsCheck(newInTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t xTensorM = OperationUtil::GetXTensorM(inTensorDescs.at(0));
    if (xTensorM * param_.twoDimTPInfo.agDim % param_.twoDimTPInfo.rsDim != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 m [" << xTensorM << "] times agDim ["
                       << param_.twoDimTPInfo.agDim
                       << "] should be an integer multiple of rsDim :" << param_.twoDimTPInfo.rsDim;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                               const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    SVector<TensorDesc> outTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(outTensors, outTensorDescs);

    switch (param_.type) {
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE:
        case atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR:
            return SetupCheckLinearAllReduce(inTensorDescs, outTensorDescs.at(0));
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            return SetupCheckLinearReduceScatter(inTensorDescs, outTensorDescs.at(0));
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            return SetupCheckAllGatherLinear(inTensorDescs, outTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR_REDUCE_SCATTER:
            return SetupCheckAllGatherLinearReduceScatter(inTensorDescs, outTensorDescs.at(0));
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "not support type:" << param_.type;
            return ERROR_INVALID_PARAM;
    }
}

Status LinearParallelOperation::SetupCheckLinearAllReduce(const SVector<TensorDesc> &inTensorDescs,
                                                          const TensorDesc &outTensorDesc) const
{
    if (inTensorDescs.at(0).dtype == ACL_INT8 &&
        param_.quantType == atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_GROUP) {
        ATB_LOG(ERROR) << GetLogPrefix() << "In the W8A8 scenario, per_group is not supported.";
        return ERROR_INVALID_PARAM;
    }
    if (!OperationUtil::MatmulInTensorDescsCheck(inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return OperationUtil::MatmulOutTensorCheck(outTensorDesc, inTensorDescs, GetLogPrefix(), commonCheckParam_) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

Status LinearParallelOperation::SetupCheckLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs,
                                                              TensorDesc &outTensorDesc) const
{
    if (InferShapeCheckLinearReduceScatter(inTensorDescs) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    outTensorDesc.shape.dims[0] *= param_.rankSize;
    return OperationUtil::MatmulOutTensorCheck(outTensorDesc, inTensorDescs, GetLogPrefix(), commonCheckParam_) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

Status LinearParallelOperation::SetupCheckAllGatherLinear(SVector<TensorDesc> &inTensorDescs,
                                                          const SVector<TensorDesc> &outTensorDescs) const
{
    if (InferShapeCheckAllGatherLinear(inTensorDescs) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    inTensorDescs.at(0).shape.dims[0] *= param_.rankSize;
    if (!OperationUtil::MatmulOutTensorCheck(outTensorDescs.at(0), inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensorDescs.size() == OUT_TENSOR_NUM_WITH_MID) {
        if (outTensorDescs.at(1).shape.dims[0] != inTensorDescs.at(0).shape.dims[0]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor1 dim0 does not match (rankSize * inTensor0 dim0)";
        }
    }
    return NO_ERROR;
}

Status LinearParallelOperation::SetupCheckAllGatherLinearReduceScatter(SVector<TensorDesc> &inTensorDescs,
                                                                       TensorDesc &outTensorDesc) const
{
    if (InferShapeCheckAllGatherLinearReduceScatter(inTensorDescs) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    inTensorDescs.at(0).shape.dims[0] *= param_.twoDimTPInfo.agDim;
    outTensorDesc.shape.dims[0] *= param_.twoDimTPInfo.rsDim;
    return OperationUtil::MatmulOutTensorCheck(outTensorDesc, inTensorDescs, GetLogPrefix(), commonCheckParam_) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

nlohmann::json LinearParallelOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

std::shared_ptr<Runner> LinearParallelOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    if (param_.backend == "hccl" || param_.backend == "lccl") {
        return std::make_shared<LinearParallelGraphRunner>(param_, *contextBase);
    } else if (param_.backend == "lcoc") {
        return std::make_shared<LinearParallelLcocRunner>(param_);
    }
    return std::shared_ptr<Runner>();
}
} // namespace atb