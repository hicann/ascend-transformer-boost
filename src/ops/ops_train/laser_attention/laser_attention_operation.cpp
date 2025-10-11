/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "laser_attention_operation.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "laser_attention_ops_runner.h"

#define CHECK_STATUS_AND_RETURN(status)                                                                                \
    if ((status) != 0) {                                                                                               \
        return (status);                                                                                               \
    }

static constexpr uint32_t IN_TENSOR_NUM = 10;
static constexpr uint32_t OUT_TENSOR_NUM = 4;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr uint64_t DIM_NUM_2 = 2;
static constexpr uint64_t DIM_NUM_3 = 3;
static constexpr uint64_t DIM_NUM_4 = 4;
static constexpr int64_t BLOCK_SIZE_ALIGN = 256;
static constexpr int64_t DEFAULT_HEAD_DIM_VALUE = 128;
static constexpr int64_t DEEP_SEEK_Q_HEAD_DIM_VALUE = 192;
static constexpr int64_t DEEP_SEEK_K_HEAD_DIM_VALUE = 256;
static constexpr int64_t MAX_HEAD_NUM = 512;

namespace {
bool ParamCheck(const atb::train::LaserAttentionParam opParam)
{
    atb::ExternalError error;
    error.errorType = atb::ERROR_INVALID_PARAM;
    error.solutionDesc = "Please check the value of params";
    bool is910B = atb::GetSingleton<atb::Config>().Is910B();
    if (!is910B) {
        error.errorDesc = "Platform is not Atlas 800I A2 inference product, operation is not supported,";
        error.errorData = atb::OperationUtil::ConcatInfo("isA2 = ", is910B);
        error.solutionDesc = "Please check the platform.";
        ATB_LOG(ERROR) << error;
        return false;
    }
    if (opParam.headNum <= 0) {
        error.errorDesc = "param headNum should be greater than 0,";
        error.errorData = atb::OperationUtil::ConcatInfo("headNum = ", opParam.headNum);
        ATB_LOG(ERROR) << error;
        return false;
    }
    if (opParam.inputLayout != "BNSD" && opParam.inputLayout != "SBH") {
        error.errorDesc = R"(param inputLayout should be "BNSD" or "SBH",)";
        error.errorData = "inputLayout = " + opParam.inputLayout;
        ATB_LOG(ERROR) << error;
        return false;
    }
    if (opParam.scaleValue <= 0 || opParam.scaleValue > 1) {
        error.errorDesc = "param scaleValue should be in range (0, 1],";
        error.errorData = atb::OperationUtil::ConcatInfo("scaleValue = ", opParam.scaleValue);
        ATB_LOG(ERROR) << error;
        return false;
    }
    if (opParam.preTokens < 1) {
        error.errorDesc = "param preTokens should be greater than or equal to 1,";
        error.errorData = atb::OperationUtil::ConcatInfo("preTokens = ", opParam.preTokens);
        ATB_LOG(ERROR) << error;
        return false;
    }
    if (opParam.preTokens % BLOCK_SIZE_ALIGN != 0) {
        error.errorDesc = "param preTokens should be should be multiple of 256,";
        error.errorData = atb::OperationUtil::ConcatInfo("preTokens = ", opParam.preTokens);
        ATB_LOG(ERROR) << error;
        return false;
    }
    if (opParam.innerPrecise != 1) {
        error.errorDesc = "param innerPrecise should be 1,";
        error.errorData = atb::OperationUtil::ConcatInfo("innerPrecise = ", opParam.innerPrecise);
        ATB_LOG(ERROR) << error;
        return false;
    }
    return true;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(LaserAttentionOperation, train::LaserAttentionParam)

LaserAttentionOperation::LaserAttentionOperation(const train::LaserAttentionParam &param)
    : OperationBase("LaserAttentionOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("LaserAttentionOperation");
}

LaserAttentionOperation::~LaserAttentionOperation() {}

uint32_t LaserAttentionOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t LaserAttentionOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

train::LaserAttentionParam LaserAttentionOperation::GetParam() const
{
    return param_;
}

void LaserAttentionOperation::SetParam(const train::LaserAttentionParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

Status LaserAttentionOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                               SVector<TensorDesc> &outTensorDescs) const
{
    size_t outTensorDescIndex = 0;
    size_t inTensorValueDescIndex = 2;
    TensorDesc &softmaxMaxTensorDesc = outTensorDescs.at(outTensorDescIndex++);
    TensorDesc &softmaxSumTensorDesc = outTensorDescs.at(outTensorDescIndex++);
    TensorDesc &softmaxOutTensorDesc = outTensorDescs.at(outTensorDescIndex++);
    TensorDesc &attentionOutTensorDesc = outTensorDescs.at(outTensorDescIndex++);
    softmaxMaxTensorDesc = inTensorDescs.at(0);
    softmaxMaxTensorDesc.dtype = ACL_FLOAT;
    softmaxMaxTensorDesc.shape.dimNum = DIM_NUM_3;
    softmaxSumTensorDesc = inTensorDescs.at(0);
    softmaxSumTensorDesc.dtype = ACL_FLOAT;
    softmaxSumTensorDesc.shape.dimNum = DIM_NUM_3;
    softmaxOutTensorDesc = inTensorDescs.at(0);
    attentionOutTensorDesc = inTensorDescs.at(0);
    if (param_.inputLayout == "BNSD") {
        attentionOutTensorDesc.shape.dims[DIM_3] = inTensorDescs.at(inTensorValueDescIndex).shape.dims[DIM_3];
    } else {
        size_t dimIndex = 0;
        int64_t dim0 = inTensorDescs.at(0).shape.dims[dimIndex++];
        int64_t dim1 = inTensorDescs.at(0).shape.dims[dimIndex++];
        int64_t dim2 = inTensorDescs.at(0).shape.dims[dimIndex++];
        size_t softmaxMaxDimIndex = 0;
        softmaxMaxTensorDesc.shape.dims[softmaxMaxDimIndex++] = dim1;
        softmaxMaxTensorDesc.shape.dims[softmaxMaxDimIndex++] = dim2 / DEEP_SEEK_Q_HEAD_DIM_VALUE;
        softmaxMaxTensorDesc.shape.dims[softmaxMaxDimIndex++] = dim0;
        size_t softmaxSumDimIndex = 0;
        softmaxSumTensorDesc.shape.dims[softmaxSumDimIndex++] = dim1;
        softmaxSumTensorDesc.shape.dims[softmaxSumDimIndex++] = dim2 / DEEP_SEEK_Q_HEAD_DIM_VALUE;
        softmaxSumTensorDesc.shape.dims[softmaxSumDimIndex++] = dim0;
        attentionOutTensorDesc.shape.dims[DIM_2] = dim2 / DEEP_SEEK_Q_HEAD_DIM_VALUE * DEFAULT_HEAD_DIM_VALUE;
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    TensorDims dims;
    if (param_.inputLayout == "BNSD") {
        return InTensorDescsCheck(inTensorDescs, dims);
    }
    if (param_.inputLayout == "SBH") {
        return InTensorDescsCheckSBH(inTensorDescs, dims);
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                               const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    TensorDims dims;
    if (param_.inputLayout == "BNSD") {
        Status status = InTensorDescsCheck(inTensorDescs, dims);
        CHECK_STATUS_AND_RETURN(status)
        status = OutTensorCheck(outTensors, dims);
        return status;
    }
    if (param_.inputLayout == "SBH") {
        Status status = InTensorDescsCheckSBH(inTensorDescs, dims);
        CHECK_STATUS_AND_RETURN(status)
        status = OutTensorCheckSBH(outTensors, dims);
        return status;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> LaserAttentionOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_LASER_ATTENTION);
    Runner *runner = pool.MallocRunner<LaserAttentionOpsRunner, train::LaserAttentionParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<LaserAttentionOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

nlohmann::json LaserAttentionOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

SVector<bool> LaserAttentionOperation::GetEmptyInTensorPermissions() const
{
    SVector<bool> emptyInTensorPermissions;
    emptyInTensorPermissions.resize(GetInputNum());
    size_t inTensorIndex = 0;
    emptyInTensorPermissions.at(inTensorIndex++) = false; // query
    emptyInTensorPermissions.at(inTensorIndex++) = false; // key
    emptyInTensorPermissions.at(inTensorIndex++) = false; // value
    emptyInTensorPermissions.at(inTensorIndex++) = true;  // pseShift
    emptyInTensorPermissions.at(inTensorIndex++) = true;  // dropMask
    emptyInTensorPermissions.at(inTensorIndex++) = true;  // paddingMask
    emptyInTensorPermissions.at(inTensorIndex++) = true;  // attenMask
    emptyInTensorPermissions.at(inTensorIndex++) = true;  // prefix
    emptyInTensorPermissions.at(inTensorIndex++) = true;  // actualSeqQLen
    emptyInTensorPermissions.at(inTensorIndex++) = true;  // actualSeqKVLen
    return emptyInTensorPermissions;
}

Status LaserAttentionOperation::InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs, TensorDims &dims) const
{
    size_t inTensorDescIndex = 0;
    const TensorDesc &queryTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &keyTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &valueTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &pseShiftTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &dropMaskTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &paddingMaskTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &attenMaskTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &prefixTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &actualSeqQLenTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &actualSeqKVLemTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    Status status = QueryTensorDescCheck(queryTensorDesc, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = KeyTensorDescCheck(keyTensorDesc, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = ValueTensorDescCheck(valueTensorDesc, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = PseShiftTensorDescCheck(pseShiftTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = DropMaskTensorDescCheck(dropMaskTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = PaddingMaskTensorDescCheck(paddingMaskTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = AttenMaskTensorDescCheck(attenMaskTensorDesc, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = PrefixTensorDescCheck(prefixTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = ActualSeqQLenTensorDescCheck(actualSeqQLenTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = ActualSeqKVLenTensorDescCheck(actualSeqKVLemTensorDesc);
    return status;
}
Status LaserAttentionOperation::InTensorDescsCheckSBH(const SVector<TensorDesc> &inTensorDescs, TensorDims &dims) const
{
    size_t inTensorDescIndex = 0;
    const TensorDesc &queryTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &keyTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &valueTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &pseShiftTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &dropMaskTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &paddingMaskTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &attenMaskTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &prefixTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &actualSeqQLenTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    const TensorDesc &actualSeqKVLemTensorDesc = inTensorDescs.at(inTensorDescIndex++);
    Status status = QueryTensorDescCheckSBH(queryTensorDesc, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = KeyTensorDescCheckSBH(keyTensorDesc, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = ValueTensorDescCheckSBH(valueTensorDesc, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = PseShiftTensorDescCheck(pseShiftTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = DropMaskTensorDescCheck(dropMaskTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = PaddingMaskTensorDescCheck(paddingMaskTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = AttenMaskTensorDescCheckSBH(attenMaskTensorDesc, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = PrefixTensorDescCheck(prefixTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = ActualSeqQLenTensorDescCheck(actualSeqQLenTensorDesc);
    CHECK_STATUS_AND_RETURN(status)
    status = ActualSeqKVLenTensorDescCheck(actualSeqKVLemTensorDesc);
    return status;
}

Status LaserAttentionOperation::OutTensorCheck(const SVector<Tensor> &outTensors, const TensorDims &dims) const
{
    size_t outTensorIndex = 0;
    const Tensor &softmaxMaxTensor = outTensors.at(outTensorIndex++);
    const Tensor &softmaxSumTensor = outTensors.at(outTensorIndex++);
    const Tensor &softmaxOutTensor = outTensors.at(outTensorIndex++);
    const Tensor &attentionOutTensor = outTensors.at(outTensorIndex++);
    Status status = SoftmaxMaxTensorCheck(softmaxMaxTensor, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = SoftmaxSumTensorCheck(softmaxSumTensor, softmaxMaxTensor);
    CHECK_STATUS_AND_RETURN(status)
    status = SoftmaxOutTensorCheck(softmaxOutTensor);
    CHECK_STATUS_AND_RETURN(status)
    status = AttentionOutTensorCheck(attentionOutTensor, dims);
    return status;
}

Status LaserAttentionOperation::OutTensorCheckSBH(const SVector<Tensor> &outTensors, const TensorDims &dims) const
{
    size_t outTensorIndex = 0;
    const Tensor &softmaxMaxTensor = outTensors.at(outTensorIndex++);
    const Tensor &softmaxSumTensor = outTensors.at(outTensorIndex++);
    const Tensor &softmaxOutTensor = outTensors.at(outTensorIndex++);
    const Tensor &attentionOutTensor = outTensors.at(outTensorIndex++);
    Status status = SoftmaxMaxTensorCheckSBH(softmaxMaxTensor, dims);
    CHECK_STATUS_AND_RETURN(status)
    status = SoftmaxSumTensorCheck(softmaxSumTensor, softmaxMaxTensor);
    CHECK_STATUS_AND_RETURN(status)
    status = SoftmaxOutTensorCheck(softmaxOutTensor);
    CHECK_STATUS_AND_RETURN(status)
    status = AttentionOutTensorCheckSBH(attentionOutTensor, dims);
    return status;
}

Status LaserAttentionOperation::QueryTensorDescCheck(const TensorDesc &queryTensorDesc, TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(queryTensorDesc, DIM_NUM_4)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor0 dimNum should be 4,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dimNum = ", queryTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    dims.batch = queryTensorDesc.shape.dims[dimIndex++];
    dims.qHeadNum = queryTensorDesc.shape.dims[dimIndex++];
    dims.seqSize = queryTensorDesc.shape.dims[dimIndex++];
    dims.qHeadDim = queryTensorDesc.shape.dims[dimIndex++];
    if (dims.qHeadNum != param_.headNum) {
        error.errorDesc = "inTensor0 dim1 and param headNum should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dim1 = ", dims.qHeadNum, ", headNum = ", param_.headNum);
        error.solutionDesc = "Please check value of params and shape of inTensors";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.qHeadNum > MAX_HEAD_NUM) {
        error.errorDesc = "Only support headNum no more than 512; inTensor0 dim1 should be less than or equal to 512,";
        error.errorData = OperationUtil::ConcatInfo("Current inTensor0 dim1 = ", dims.qHeadNum);
        error.solutionDesc = "Please check value of params and shape of inTensors";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.seqSize % BLOCK_SIZE_ALIGN != 0) {
        error.errorDesc = "inTensor0 dim2 should be multiple of 256,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dim2 = ", dims.seqSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.qHeadDim == DEFAULT_HEAD_DIM_VALUE) {
        if (dims.seqSize < param_.preTokens) {
            error.errorDesc = "inTensor0 dim3 is 128, inTensor0 dim2 should not be less than param preTokens,";
            error.errorData =
                OperationUtil::ConcatInfo("inTensor0 dim2 = ", dims.seqSize, ", preTokens = ", param_.preTokens);
            error.solutionDesc = "Please check value of params and shape of inTensors";
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return error.errorType;
        }
    } else if (dims.qHeadDim == DEEP_SEEK_Q_HEAD_DIM_VALUE) {
        if (dims.seqSize != param_.preTokens) {
            error.errorDesc = "inTensor0 dim3 is 192, inTensor0 dim2 and param preTokens should be equal,";
            error.errorData =
                OperationUtil::ConcatInfo("inTensor0 dim2 = ", dims.seqSize, ", preTokens = ", param_.preTokens);
            error.solutionDesc = "Please check value of params and shape of inTensors";
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return error.errorType;
        }
    } else {
        error.errorDesc = "inTensor0 dim3 should be 128 or 192,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dim3 = ", dims.qHeadDim);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}
Status LaserAttentionOperation::QueryTensorDescCheckSBH(const TensorDesc &queryTensorDesc, TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(queryTensorDesc, DIM_NUM_3)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor0 dimNum should be 3,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dimNum = ", queryTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    dims.seqSize = queryTensorDesc.shape.dims[dimIndex++];
    dims.batch = queryTensorDesc.shape.dims[dimIndex++];
    dims.qHeadDim = DEEP_SEEK_Q_HEAD_DIM_VALUE;
    dims.qHeadNum = queryTensorDesc.shape.dims[dimIndex++] / dims.qHeadDim;
    if (dims.seqSize % BLOCK_SIZE_ALIGN != 0) {
        error.errorDesc = "inTensor0 dim0 should be multiple of 256,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dim0 = ", dims.seqSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.seqSize != param_.preTokens) {
        error.errorDesc = "inTensor0 dim0 and param preTokens should be equal,";
        error.errorData =
            OperationUtil::ConcatInfo("inTensor0 dim0 = ", dims.seqSize, ", preTokens = ", param_.preTokens);
        error.solutionDesc = "Please check value of params and shape of inTensors";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.qHeadNum != param_.headNum) {
        error.errorDesc = "inTensor0 dim2 divided by 192 and param headNum should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dim2 = ", dims.qHeadNum * dims.qHeadDim,
                                                    ", headNum = ", param_.headNum);
        error.solutionDesc = "Please check value of params and shape of inTensors";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.qHeadNum > MAX_HEAD_NUM) {
        error.errorDesc =
            "Only support headNum no more than 512; inTensor0 dim2 divided by 192 should be less than or equal to 512,";
        error.errorData = OperationUtil::ConcatInfo("Current inTensor0 dim2 = ", dims.qHeadNum * dims.qHeadDim);
        error.solutionDesc = "Please check value of params and shape of inTensors";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::KeyTensorDescCheck(const TensorDesc &keyTensorDesc, TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(keyTensorDesc, DIM_NUM_4)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor1 dimNum should be 4,";
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dimNum = ", keyTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    int64_t dim0 = keyTensorDesc.shape.dims[dimIndex++];
    dims.kvHeadNum = keyTensorDesc.shape.dims[dimIndex++];
    dims.kvSize = keyTensorDesc.shape.dims[dimIndex++];
    dims.kHeadDim = keyTensorDesc.shape.dims[dimIndex++];
    if (dim0 != dims.batch) {
        error.errorDesc = "inTensor1 dim0 and inTensor0 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dim0 = ", dim0, ", inTensor0 dim0 = ", dims.batch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.qHeadNum % dims.kvHeadNum != 0) {
        error.errorDesc = "inTensor0 dim1 should be multiple of inTensor1 dim1,";
        error.errorData =
            OperationUtil::ConcatInfo("inTensor0 dim1 = ", dims.qHeadNum, ", inTensor1 dim1 = ", dims.kvHeadNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.kvSize % BLOCK_SIZE_ALIGN != 0) {
        error.errorDesc = "inTensor1 dim2 should be multiple of 256,";
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dim2 = ", dims.kvSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    int64_t kHeadDim = dims.qHeadDim == DEFAULT_HEAD_DIM_VALUE ? DEFAULT_HEAD_DIM_VALUE : DEEP_SEEK_K_HEAD_DIM_VALUE;
    if ((dims.qHeadDim == DEFAULT_HEAD_DIM_VALUE && dims.kHeadDim != DEFAULT_HEAD_DIM_VALUE) ||
        (dims.qHeadDim == DEEP_SEEK_Q_HEAD_DIM_VALUE && dims.kHeadDim != DEEP_SEEK_K_HEAD_DIM_VALUE)) {
        error.errorDesc =
            OperationUtil::ConcatInfo("inTensor0 dim3 = ", dims.qHeadDim, ", inTensor1 dim3 should be ", kHeadDim, ",");
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dim3 = ", dims.kHeadDim);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::KeyTensorDescCheckSBH(const TensorDesc &keyTensorDesc, TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(keyTensorDesc, DIM_NUM_3)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor1 dimNum should be 3,";
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dimNum = ", keyTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    dims.kvSize = keyTensorDesc.shape.dims[dimIndex++];
    int64_t dim1 = keyTensorDesc.shape.dims[dimIndex++];
    dims.kHeadDim = DEEP_SEEK_K_HEAD_DIM_VALUE;
    dims.kvHeadNum = keyTensorDesc.shape.dims[dimIndex++] / dims.kHeadDim;
    if (dims.kvSize % BLOCK_SIZE_ALIGN != 0) {
        error.errorDesc = "inTensor1 dim0 should be multiple of 256,";
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dim0 = ", dims.kvSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim1 != dims.batch) {
        error.errorDesc = "inTensor1 dim1 and inTensor0 dim1 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dim1 = ", dim1, ", inTensor0 dim1 = ", dims.batch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.qHeadNum % dims.kvHeadNum != 0) {
        error.errorDesc = "inTensor0 dim2 divided by 192 should be multiple of inTensor1 dim2 divided by 256,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dim2 = ", dims.qHeadNum * dims.qHeadDim,
                                                    ", inTensor1 dim2 = ", dims.kvHeadNum * dims.kHeadDim);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::ValueTensorDescCheck(const TensorDesc &valueTensorDesc, TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(valueTensorDesc, DIM_NUM_4)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor2 dimNum should be 4,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dimNum = ", valueTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    int64_t dim0 = valueTensorDesc.shape.dims[dimIndex++];
    int64_t dim1 = valueTensorDesc.shape.dims[dimIndex++];
    int64_t dim2 = valueTensorDesc.shape.dims[dimIndex++];
    dims.vHeadDim = valueTensorDesc.shape.dims[dimIndex++];
    if (dim0 != dims.batch) {
        error.errorDesc = "inTensor2 dim0 and inTensor0 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dim0 = ", dim0, ", inTensor0 dim0 = ", dims.batch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim1 != dims.kvHeadNum) {
        error.errorDesc = "inTensor2 dim1 and inTensor1 dim1 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dim1 = ", dim1, ", inTensor1 dim1 = ", dims.kvHeadNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim2 != dims.kvSize) {
        error.errorDesc = "inTensor2 dim2 and inTensor1 dim2 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dim2 = ", dim2, ", inTensor1 dim2 = ", dims.kvSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.vHeadDim != DEFAULT_HEAD_DIM_VALUE) {
        error.errorDesc = "inTensor2 dim3 should be 128,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dim3 = ", dims.vHeadDim);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}
Status LaserAttentionOperation::ValueTensorDescCheckSBH(const TensorDesc &valueTensorDesc, TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(valueTensorDesc, DIM_NUM_3)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor2 dimNum should be 3,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dimNum = ", valueTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    int64_t dim0 = valueTensorDesc.shape.dims[dimIndex++];
    int64_t dim1 = valueTensorDesc.shape.dims[dimIndex++];
    int64_t dim2 = valueTensorDesc.shape.dims[dimIndex++];
    dims.vHeadDim = DEFAULT_HEAD_DIM_VALUE;
    if (dim0 != dims.kvSize) {
        error.errorDesc = "inTensor2 dim0 and inTensor1 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dim0 = ", dim0, ", inTensor1 dim0 = ", dims.kvSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim1 != dims.batch) {
        error.errorDesc = "inTensor2 dim1 and inTensor0 dim1 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dim1 = ", dim1, ", inTensor0 dim1 = ", dims.batch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim2 / dims.vHeadDim != dims.kvHeadNum) {
        error.errorDesc = "inTensor2 dim2 divided by 128 and inTensor1 dim2 divided by 256 should be equal,";
        error.errorData =
            OperationUtil::ConcatInfo("inTensor2 dim2 = ", dim2, ", inTensor1 dim2 = ", dims.kvHeadNum * dims.kHeadDim);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::PseShiftTensorDescCheck(const TensorDesc &pseShiftTensorDesc) const
{
    if (TensorCheck::IsEmptyTensor(pseShiftTensorDesc)) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    error.errorDesc = "inTensor3 should be empty tensor,";
    error.errorData = OperationUtil::ConcatInfo("inTensor3 dimNum = ", pseShiftTensorDesc.shape.dimNum);
    error.solutionDesc = "Please check shape of inTensors";
    return ERROR_INVALID_TENSOR_DIM_NUM;
}

Status LaserAttentionOperation::DropMaskTensorDescCheck(const TensorDesc &dropMaskTensorDesc) const
{
    if (TensorCheck::IsEmptyTensor(dropMaskTensorDesc)) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    error.errorDesc = "inTensor4 should be empty tensor,";
    error.errorData = OperationUtil::ConcatInfo("inTensor4 dimNum = ", dropMaskTensorDesc.shape.dimNum);
    error.solutionDesc = "Please check shape of inTensors";
    return ERROR_INVALID_TENSOR_DIM_NUM;
}

Status LaserAttentionOperation::PaddingMaskTensorDescCheck(const TensorDesc &paddingMaskTensorDesc) const
{
    if (TensorCheck::IsEmptyTensor(paddingMaskTensorDesc)) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    error.errorDesc = "inTensor5 should be empty tensor,";
    error.errorData = OperationUtil::ConcatInfo("inTensor5 dimNum = ", paddingMaskTensorDesc.shape.dimNum);
    error.solutionDesc = "Please check shape of inTensors";
    return ERROR_INVALID_TENSOR_DIM_NUM;
}

Status LaserAttentionOperation::AttenMaskTensorDescCheck(const TensorDesc &attenMaskTensorDesc,
                                                         const TensorDims &dims) const
{
    if (TensorCheck::IsEmptyTensor(attenMaskTensorDesc)) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(attenMaskTensorDesc, DIM_NUM_2)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor6 dimNum should be 2,";
        error.errorData = OperationUtil::ConcatInfo("inTensor6 dimNum = ", attenMaskTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.seqSize != dims.kvSize) {
        error.errorDesc = "inTensor6 is not empty tensor, inTensor0 dim2 and inTensor1 dim2 should be equal,";
        error.errorData =
            OperationUtil::ConcatInfo("inTensor6 dimNum = ", attenMaskTensorDesc.shape.dimNum,
                                      ", inTensor0 dim2 = ", dims.seqSize, ", inTensor1 dim2 = ", dims.kvSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (attenMaskTensorDesc.shape.dims[0] != dims.seqSize) {
        error.errorDesc = "inTensor6 dim0 and inTensor0 dim2 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor6 dim0 = ", attenMaskTensorDesc.shape.dims[0],
                                                    ", inTensor0 dim2 = ", dims.seqSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (attenMaskTensorDesc.shape.dims[1] != dims.kvSize) {
        error.errorDesc = "inTensor6 dim1 and inTensor1 dim2 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor6 dim1 = ", attenMaskTensorDesc.shape.dims[1],
                                                    ", inTensor1 dim2 = ", dims.kvSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::AttenMaskTensorDescCheckSBH(const TensorDesc &attenMaskTensorDesc,
                                                            const TensorDims &dims) const
{
    if (TensorCheck::IsEmptyTensor(attenMaskTensorDesc)) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(attenMaskTensorDesc, DIM_NUM_2)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor6 dimNum should be 2,";
        error.errorData = OperationUtil::ConcatInfo("inTensor6 dimNum = ", attenMaskTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dims.seqSize != dims.kvSize) {
        error.errorDesc = "inTensor6 is not empty tensor, inTensor0 dim0 and inTensor1 dim0 should be equal,";
        error.errorData =
            OperationUtil::ConcatInfo("inTensor6 dimNum = ", attenMaskTensorDesc.shape.dimNum,
                                      ", inTensor0 dim0 = ", dims.seqSize, ", inTensor1 dim0 = ", dims.kvSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (attenMaskTensorDesc.shape.dims[0] != dims.seqSize) {
        error.errorDesc = "inTensor6 dim0 and inTensor0 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor6 dim0 = ", attenMaskTensorDesc.shape.dims[0],
                                                    ", inTensor0 dim0 = ", dims.seqSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (attenMaskTensorDesc.shape.dims[1] != dims.kvSize) {
        error.errorDesc = "inTensor6 dim1 and inTensor1 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor6 dim1 = ", attenMaskTensorDesc.shape.dims[1],
                                                    ", inTensor1 dim0 = ", dims.kvSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::PrefixTensorDescCheck(const TensorDesc &prefixTensorDesc) const
{
    if (TensorCheck::IsEmptyTensor(prefixTensorDesc)) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    error.errorDesc = "inTensor7 should be empty tensor,";
    error.errorData = OperationUtil::ConcatInfo("inTensor7 dimNum = ", prefixTensorDesc.shape.dimNum);
    error.solutionDesc = "Please check shape of inTensors";
    return ERROR_INVALID_TENSOR_DIM_NUM;
}

Status LaserAttentionOperation::ActualSeqQLenTensorDescCheck(const TensorDesc &actualSeqQLenTensorDesc) const
{
    if (TensorCheck::IsEmptyTensor(actualSeqQLenTensorDesc)) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    error.errorDesc = "inTensor8 should be empty tensor,";
    error.errorData = OperationUtil::ConcatInfo("inTensor8 dimNum = ", actualSeqQLenTensorDesc.shape.dimNum);
    error.solutionDesc = "Please check shape of inTensors";
    return ERROR_INVALID_TENSOR_DIM_NUM;
}

Status LaserAttentionOperation::ActualSeqKVLenTensorDescCheck(const TensorDesc &actualSeqKVLenTensorDesc) const
{
    if (TensorCheck::IsEmptyTensor(actualSeqKVLenTensorDesc)) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    error.errorDesc = "inTensor9 should be empty tensor,";
    error.errorData = OperationUtil::ConcatInfo("inTensor9 dimNum = ", actualSeqKVLenTensorDesc.shape.dimNum);
    error.solutionDesc = "Please check shape of inTensors";
    return ERROR_INVALID_TENSOR_DIM_NUM;
}

Status LaserAttentionOperation::SoftmaxMaxTensorCheck(const Tensor &softmaxMaxTensor, const TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors and outTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(softmaxMaxTensor.desc, DIM_NUM_3)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "outTensor0 dimNum should be 3,";
        error.errorData = OperationUtil::ConcatInfo("outTensor0 dimNum = ", softmaxMaxTensor.desc.shape.dimNum);
        error.solutionDesc = "Please check shape of outTensors";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    int64_t dim0 = softmaxMaxTensor.desc.shape.dims[dimIndex++];
    int64_t dim1 = softmaxMaxTensor.desc.shape.dims[dimIndex++];
    int64_t dim2 = softmaxMaxTensor.desc.shape.dims[dimIndex++];
    if (dim0 != dims.batch) {
        error.errorDesc = "outTensor0 dim0 and inTensor0 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor0 dim0 = ", dim0, ", inTensor0 dim0 = ", dims.batch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim1 != dims.qHeadNum) {
        error.errorDesc = "outTensor0 dim1 and inTensor0 dim1 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor0 dim1 = ", dim1, ", inTensor0 dim1 = ", dims.qHeadNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim2 != dims.seqSize) {
        error.errorDesc = "outTensor0 dim2 and inTensor0 dim2 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor0 dim2 = ", dim2, ", inTensor0 dim2 = ", dims.seqSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}
Status LaserAttentionOperation::SoftmaxMaxTensorCheckSBH(const Tensor &softmaxMaxTensor, const TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors and outTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(softmaxMaxTensor.desc, DIM_NUM_3)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "outTensor0 dimNum should be 3,";
        error.errorData = OperationUtil::ConcatInfo("outTensor0 dimNum = ", softmaxMaxTensor.desc.shape.dimNum);
        error.solutionDesc = "Please check shape of outTensors";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    int64_t dim0 = softmaxMaxTensor.desc.shape.dims[dimIndex++];
    int64_t dim1 = softmaxMaxTensor.desc.shape.dims[dimIndex++];
    int64_t dim2 = softmaxMaxTensor.desc.shape.dims[dimIndex++];
    if (dim0 != dims.batch) {
        error.errorDesc = "outTensor0 dim0 and inTensor0 dim1 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor0 dim0 = ", dim0, ", inTensor0 dim1 = ", dims.batch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim1 != dims.qHeadNum) {
        error.errorDesc = "outTensor0 dim1 and inTensor0 dim2 divided by 192 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor0 dim1 = ", dim1, ", inTensor0 dim2 = ", dims.qHeadNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim2 != dims.seqSize) {
        error.errorDesc = "outTensor0 dim2 and inTensor0 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor0 dim2 = ", dim2, ", inTensor0 dim0 = ", dims.seqSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}


Status LaserAttentionOperation::SoftmaxSumTensorCheck(const Tensor &softmaxSumTensor,
                                                      const Tensor &softmaxMaxTensor) const
{
    TensorDesc softmaxSumTensorDesc = softmaxSumTensor.desc;
    TensorDesc softmaxMaxTensorDesc = softmaxMaxTensor.desc;
    Status status = TensorCheck::TensorDescsEqual(softmaxMaxTensorDesc, softmaxSumTensorDesc);
    if (status == NO_ERROR) {
        return NO_ERROR;
    }
    ExternalError error;
    error.errorType = static_cast<ErrorType>(status);
    error.errorDesc = "outTensor1 shape and outTensor0 shape should be the same,";
    error.errorData =
        OperationUtil::ConcatInfo("outTensor1 shape = ", TensorUtil::ShapeToString(softmaxSumTensorDesc.shape),
                                  ", outTensor0 shape = ", TensorUtil::ShapeToString(softmaxMaxTensorDesc.shape));
    error.solutionDesc = "Please check shape of outTensors";
    ATB_LOG(ERROR) << GetLogPrefix() << error;
    return error.errorType;
}

Status LaserAttentionOperation::SoftmaxOutTensorCheck(const Tensor &softmaxOutTensor) const
{
    (void)softmaxOutTensor;
    return NO_ERROR;
}

Status LaserAttentionOperation::AttentionOutTensorCheck(const Tensor &attentionOutTensor, const TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors and outTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(attentionOutTensor.desc, DIM_NUM_4)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "outTensor3 dimNum should be 4,";
        error.errorData = OperationUtil::ConcatInfo("outTensor3 dimNum = ", attentionOutTensor.desc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    int64_t dim0 = attentionOutTensor.desc.shape.dims[dimIndex++];
    int64_t dim1 = attentionOutTensor.desc.shape.dims[dimIndex++];
    int64_t dim2 = attentionOutTensor.desc.shape.dims[dimIndex++];
    int64_t dim3 = attentionOutTensor.desc.shape.dims[dimIndex++];
    if (dim0 != dims.batch) {
        error.errorDesc = "outTensor3 dim0 and inTensor0 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor3 dim0 = ", dim0, ", inTensor0 dim0 = ", dims.batch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim1 != dims.qHeadNum) {
        error.errorDesc = "outTensor3 dim1 and inTensor0 dim1 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor3 dim1 = ", dim1, ", inTensor0 dim1 = ", dims.qHeadNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim2 != dims.seqSize) {
        error.errorDesc = "outTensor3 dim2 and inTensor0 dim2 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor3 dim2 = ", dim2, ", inTensor0 dim2 = ", dims.seqSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim3 != dims.vHeadDim) {
        error.errorDesc = "outTensor3 dim3 should be 128,";
        error.errorData = OperationUtil::ConcatInfo("outTensor3 dim3 = ", dims.vHeadDim);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LaserAttentionOperation::AttentionOutTensorCheckSBH(const Tensor &attentionOutTensor,
                                                           const TensorDims &dims) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors and outTensors";
    if (!TensorCheck::IsTensorDescDimNumValid(attentionOutTensor.desc, DIM_NUM_3)) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "outTensor3 dimNum should be 3,";
        error.errorData = OperationUtil::ConcatInfo("outTensor3 dimNum = ", attentionOutTensor.desc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    size_t dimIndex = 0;
    int64_t dim0 = attentionOutTensor.desc.shape.dims[dimIndex++];
    int64_t dim1 = attentionOutTensor.desc.shape.dims[dimIndex++];
    int64_t dim2 = attentionOutTensor.desc.shape.dims[dimIndex++];
    if (dim0 != dims.seqSize) {
        error.errorDesc = "outTensor3 dim0 and inTensor0 dim0 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor3 dim0 = ", dim0, ", inTensor0 dim0 = ", dims.seqSize);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim1 != dims.batch) {
        error.errorDesc = "outTensor3 dim1 and inTensor0 dim1 should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor3 dim1 = ", dim1, ", inTensor0 dim1 = ", dims.batch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (dim2 / dims.vHeadDim != dims.qHeadNum) {
        error.errorDesc = "outTensor3 dim2 divided by 128 and inTensor0 dim2 divided by 192 should be equal,";
        error.errorData =
            OperationUtil::ConcatInfo("outTensor3 dim2 = ", dim2, ", inTensor0 dim2 = ", dims.qHeadNum * dims.qHeadDim);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}
} // namespace atb