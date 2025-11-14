/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "self_attention_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include <securec.h>
#include "atb/utils/dl_manager.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"
#include "acl/acl.h"

namespace {
static const int DIM0 = 0;
static const int DIM1 = 1;
static const int DIM2 = 2;
static const int DIM3 = 3;
static const int BASE_INPUT_NUM = 3;
static const int OUTPUT_NUM = 1;
static const int OUTPUT_NUM_ACLNN = 2;
static const int TENSORLIST_NUM = 2;
static const size_t ATB_MASK_INDEX = 3;
static const int ACLNN_PSEDHIFT_INDEX = 3;
static const int ACLNN_MASK_INDEX = 4;
static const int64_t INT_MAX_VALUE = 2147483647;
static const int64_t COMPRESS_MASK_SHAPE = 2048;
static const int SEQLEN_INDEX = 3;
}

namespace atb {
aclnnStatus (*SelfAttentionAclnnRunner::aclnnGetWorkspaceSizeFunc_)(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, const aclTensor *attentionOut, const aclTensor *softmaxLse,
    uint64_t *workspaceSize, aclOpExecutor **executor) = nullptr;
aclnnStatus (*SelfAttentionAclnnRunner::aclnnExecuteFunc_)(void *workspace, uint64_t workspaceSize,
                                                           aclOpExecutor *executor, const aclrtStream stream) = nullptr;

SelfAttentionAclnnRunner::SelfAttentionAclnnRunner(const atb::infer::SelfAttentionParam &opParam)
    : AclnnRunner("SelfAttentionAclnnRunner"), param_(opParam)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SelfAttentionAclnnRunner::SelfAttentionAclnnRunner called";
}

SelfAttentionAclnnRunner::~SelfAttentionAclnnRunner() {}

Status SelfAttentionAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    ret = CreateAclnnInTensor(runnerVariantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << "SelfAttentionAclnnRunner CreateAclnnInTensor fail";
        return atb::ERROR_INVALID_PARAM;
    }

    ret = CreateAclnnInTensorList(runnerVariantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << "SelfAttentionAclnnRunner CreateAclnnInTensorList fail";
        return atb::ERROR_INVALID_PARAM;
    }

    ret = CreateAclnnOutTensor(runnerVariantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << "SelfAttentionAclnnRunner CreateAclNNOutTensorVariantPack fail";
        return atb::ERROR_INVALID_PARAM;
    }

    return atb::NO_ERROR;
}

uint32_t SelfAttentionAclnnRunner::GetInputNumAclnn() const
{
    int inputNum = BASE_INPUT_NUM;
    if (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        inputNum++;
    }
    return inputNum; // SelfAttention入参个数
}

uint32_t SelfAttentionAclnnRunner::GetOutputNumAclnn() const
{
    return OUTPUT_NUM_ACLNN; // FIA出参个数
}

uint32_t SelfAttentionAclnnRunner::GetInputNumAtb() const
{
    return GetInputNumAclnn() + 1; // 1 : seqLen
}

uint32_t SelfAttentionAclnnRunner::GetOutputNumAtb() const
{
    return OUTPUT_NUM; // SelfAttentionPluginOperation出参个数
}

atb::Status SelfAttentionAclnnRunner::CreateAclnnInTensor(const RunnerVariantPack &runnerVariantPack)
{
    this->aclnnVariantPack_.aclInTensors.reserve(GetInputNumAclnn());
    this->aclnnVariantPack_.aclInTensors.resize(GetInputNumAclnn());
    Status ret = NO_ERROR;
    int actualSeqLenIndex = SEQLEN_INDEX;
    int maskIndex = -1;
    if (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        actualSeqLenIndex++;
        maskIndex = ATB_MASK_INDEX;
    }
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "SelfAttentionAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        int index = i;
        if (index == maskIndex) {
            index = ACLNN_MASK_INDEX;
        }
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(index);
        bool reallyNoNeedMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED && q_S_ == 1);
        aclnnTensorPtr->needUpdateTensorDataPtr = (reallyNoNeedMask && i == ATB_MASK_INDEX) ? false : true;
        this->aclnnVariantPack_.aclInTensors.at(i) = aclnnTensorPtr;
    }
    ret = ProcessSeqLengthTensor(runnerVariantPack.inTensors.at(actualSeqLenIndex));
    if (ret != atb::NO_ERROR) {
        return ret;
    }
    return atb::NO_ERROR;
}

atb::Status SelfAttentionAclnnRunner::CreateAclnnInTensorList(const RunnerVariantPack &runnerVariantPack)
{
    int kvTensorNum = 1;
    int tensorListOffset = 1;
    this->aclnnVariantPack_.aclInTensorList.reserve(TENSORLIST_NUM);
    this->aclnnVariantPack_.aclInTensorList.resize(TENSORLIST_NUM);
    Status ret = NO_ERROR;
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensorList.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i + tensorListOffset);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;

        aclTensor *aclTensorKV = aclnnTensorPtr->tensor;
        if (aclTensorKV == nullptr) {
            ATB_LOG(ERROR) << " InTensor aclCreateTensor index " << std::to_string(i) << " fail";
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclTensor *tensorsOfKV[kvTensorNum];
        tensorsOfKV[0] = aclTensorKV;
        auto tensorValueList = aclCreateTensorList(tensorsOfKV, kvTensorNum);
        this->aclnnVariantPack_.aclInTensorList[i] = tensorValueList;
    }
    return atb::NO_ERROR;
}

atb::Status SelfAttentionAclnnRunner::CreateAclnnOutTensor(const RunnerVariantPack &runnerVariantPack)
{
    this->aclnnVariantPack_.aclOutTensors.reserve(GetOutputNumAclnn());
    this->aclnnVariantPack_.aclOutTensors.resize(GetOutputNumAclnn());
    Status ret = NO_ERROR;
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    ATB_LOG(INFO) << GetLogPrefix() << "SelfAttentionAclnnRunner::BuildAclnnVariantPack outTensor index: " << 0;
    atb::Tensor atbTensor = runnerVariantPack.outTensors.at(0);
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                              atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    aclnnTensorPtr->tensorIdx = static_cast<int>(0);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclOutTensors.at(0) = aclnnTensorPtr;
    this->aclnnVariantPack_.aclOutTensors.at(1) = aclnnTensorPtr;
    return atb::NO_ERROR;
}

atb::Status SelfAttentionAclnnRunner::ProcessSeqLengthTensor(const atb::Tensor tensor)
{
    ATB_LOG(INFO) << " ProcessSeqLengthTensor start";
    int dataSize = tensor.dataSize / 4; // 4: int32 size
    if (seqLencache_.size() == 0) {
        seqLencache_.reserve(dataSize);
        seqLencache_.resize(dataSize);
        seqLen_.reserve(dataSize);
        seqLen_.resize(dataSize);
    }
    if (tensor.hostData == nullptr) {
        ATB_LOG(ERROR) << "Host data is null";
        return atb::ERROR_INVALID_TENSOR_ADDR;
    }
    if (memcpy_s(seqLencache_.data(), dataSize * 4, tensor.hostData, dataSize * 4) != 0) { // 4: int32 size
        ATB_LOG(ERROR) << " Get seqLen failed.";
        return atb::ERROR_INTERNAL_ERROR;
    }
    for (int j = 0; j < dataSize; ++j) {
        if (j == 0) {
            seqLen_[j] = static_cast<int64_t>(seqLencache_[j]);
        } else {
            seqLen_[j] = static_cast<int64_t>(seqLencache_[j]) + seqLen_[j - 1]; // from seqLen to prefixsum seqlen
        }
    }
    if (actualSeqLengths_ != nullptr) {
        aclDestroyIntArray(actualSeqLengths_);
        actualSeqLengths_ = nullptr;
    }
    actualSeqLengths_ = aclCreateIntArray(static_cast<int64_t *>(seqLen_.data()), dataSize);
    return atb::NO_ERROR;
}

Status SelfAttentionAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner LoadMethod";
    if (SelfAttentionAclnnRunner::aclnnGetWorkspaceSizeFunc_ &&
        SelfAttentionAclnnRunner::aclnnExecuteFunc_) {
        return NO_ERROR;
    }
    static DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
    Status ret = dlManager.getSymbol("aclnnFusedInferAttentionScoreV3GetWorkspaceSize",
                                     (void *&)SelfAttentionAclnnRunner::aclnnGetWorkspaceSizeFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnFusedInferAttentionScoreV3GetWorkspaceSize failed! Consider upgrade the CANN first!";
        return ret;
    }
    ret = dlManager.getSymbol("aclnnFusedInferAttentionScoreV3", (void *&)SelfAttentionAclnnRunner::aclnnExecuteFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnFusedInferAttentionScoreV3 failed! Consider upgrade the CANN first!";
        return ret;
    }
    ATB_LOG(INFO) << "load aclnnFusedInferAttentionScoreV3 two-staged method success!";
    return NO_ERROR;
}

aclnnStatus SelfAttentionAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn FusedInferAttentionScoreV3 setup start.";
    Status loadStatus = SelfAttentionAclnnRunner::LoadMethod();
    if (loadStatus != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    if (!aclnnGetWorkspaceSizeFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    ATB_LOG(INFO) << " aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << raw_executor_ptr
                  << ", then take the address of the raw ptr: " << &raw_executor_ptr;

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);
    AclnnSelfAttentionParam aclnnFIAParam;
    auto ret = ParamTransfer(aclnnFIAParam);
    if (ret != atb::ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << " ParamTransfer failed, ret: " << std::to_string(ret);
        return ret;
    }
    auto layerOut = std::make_unique<char[]>(aclnnFIAParam.inputLayout.length() + 1);
    errno_t result =
        strcpy_s(layerOut.get(), aclnnFIAParam.inputLayout.length() + 1, aclnnFIAParam.inputLayout.c_str());
    if (result != 0) {
        ATB_LOG(ERROR) << " inputLayout init failed, ret:" << result;
        return ERROR_INVALID_PARAM;
    }
    bool needMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED && q_S_ != 1);

    aclnnStatus aclnnRet = SelfAttentionAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        this->aclnnVariantPack_.aclInTensors.at(0)->tensor,                                   // queryTensor
        this->aclnnVariantPack_.aclInTensorList.at(0),                                        // tensorKeyList
        this->aclnnVariantPack_.aclInTensorList.at(1),                                        // tensorValueList
        nullptr,                                                                              // pseShift
        needMask ? this->aclnnVariantPack_.aclInTensors.at(ATB_MASK_INDEX)->tensor : nullptr, // attenMask
        actualSeqLengths_,                                                                    // actualSeqLengths
        actualSeqLengths_,                                                                    // actualSeqLengthsKv
        nullptr,                                                                              // deqScale1
        nullptr,                                                                              // quantScale1
        nullptr,                                                                              // deqScale2
        nullptr,                                                                              // quantScale2
        nullptr,                                                                              // quantOffset2
        nullptr,                                                                              // antiquantScale
        nullptr,                                                                              // antiquantOffset
        nullptr,                                                                              // blockTable
        nullptr,                                                                              // queryPaddingSize
        nullptr,                                                                              // kvPaddingSize
        nullptr,                                                                              // keyAntiquantScale
        nullptr,                                                                              // keyAntiquantOffset
        nullptr,                                                                              // valueAntiquantScale
        nullptr,                                                                              // valueAntiquantOffset
        nullptr,                                                                              // keySharedPrefix
        nullptr,                                                                              // valueSharedPrefix
        nullptr,                                                                              // actualSharedPrefixLen
        nullptr,                                                                              // queryRope
        nullptr,                                                                              // keyRope
        nullptr,                                                                              // keyRopeAntiquantScale
        aclnnFIAParam.numHeads, aclnnFIAParam.scaleValue, aclnnFIAParam.preTokens, aclnnFIAParam.nextTokens,
        layerOut.get(), aclnnFIAParam.numKeyValueHeads, aclnnFIAParam.sparseMode, aclnnFIAParam.innerPrecise,
        aclnnFIAParam.blockSize, aclnnFIAParam.antiquantMode, aclnnFIAParam.softmaxLseFlag,
        aclnnFIAParam.keyAntiquantMode, aclnnFIAParam.valueAntiquantMode,
        this->aclnnVariantPack_.aclOutTensors.at(0)->tensor, // outTensor
        this->aclnnVariantPack_.aclOutTensors.at(1)->tensor, // nullptr
        &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return aclnnRet;
}

Status SelfAttentionAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    Status loadStatus = SelfAttentionAclnnRunner::LoadMethod();
    if (loadStatus != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "load getWorkspace function from aclnn failed! Consider upgrade CANN first!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    if (!aclnnExecuteFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ERROR_INVALID_PARAM;
    }
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = SelfAttentionAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

atb::Status SelfAttentionAclnnRunner::ParamTransfer(AclnnSelfAttentionParam &targetParam)
{
    targetParam.numHeads = param_.headNum;
    targetParam.scaleValue = param_.qkScale;
    targetParam.preTokens = INT_MAX_VALUE;
    targetParam.nextTokens = INT_MAX_VALUE;
    std::string inputLayoutStr;
    switch (param_.inputLayout) {
        case atb::infer::InputLayout::TYPE_BSND:
            inputLayoutStr = "TND";
            break;
        case atb::infer::InputLayout::TYPE_BNSD:
            inputLayoutStr = "BNSD";
            break;
        case atb::infer::InputLayout::TYPE_BSH:
            inputLayoutStr = "BSH";
            break;
        case atb::infer::InputLayout::TYPE_TND:
            inputLayoutStr = "TND";
            break;
        case atb::infer::InputLayout::TYPE_BNSD_BSND:
            inputLayoutStr = "BNSD_BSND";
            break;
        default:
            ATB_LOG(ERROR) << "91095 only supports BSND and TND.";
            return atb::ERROR_INVALID_PARAM;
    }
    targetParam.inputLayout = inputLayoutStr;
    targetParam.numKeyValueHeads = param_.kvHeadNum == 0 ? param_.headNum : param_.kvHeadNum;
    bool needComMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED && q_S_ != 1 &&
                        mask_S_ == COMPRESS_MASK_SHAPE && param_.isTriuMask != 0);
    targetParam.sparseMode = needComMask ? 2 : 0;
    targetParam.innerPrecise = 1;
    targetParam.blockSize = 0;
    targetParam.antiquantMode = 0;
    targetParam.softmaxLseFlag = false;
    targetParam.keyAntiquantMode = 0;
    targetParam.valueAntiquantMode = 0;
    return atb::NO_ERROR;
}

int32_t SelfAttentionAclnnRunner::CalSeqLenSum(const SVector<Tensor> &inTensors, const size_t seqLenTensorId)
{
    const Tensor &seqLenTensor = inTensors.at(seqLenTensorId);
    if (!seqLenTensor.hostData) {
#ifdef _DEBUG
        ATB_LOG(ERROR) << "tensor.hostData is null, seqLenTensor.hostData:" << seqLenTensor.hostData;
#else
        ATB_LOG(ERROR) << "tensor.hostData is null";
#endif
        return false;
    }

    // dims = [batch]
    uint32_t batch = seqLenTensor.desc.shape.dims[0];

    int32_t seqLenSum = 0;
    int32_t *seqLenTensorHostData = (int32_t *)seqLenTensor.hostData;
    for (size_t i = 0; i < batch; ++i) {
        seqLenSum += seqLenTensorHostData[i];
    }
    ATB_LOG(INFO) << "nTokens:" << seqLenSum;
    return seqLenSum;
}

uint32_t SelfAttentionAclnnRunner::GetInputNum() const
{
    return GetInputNumAtb(); // SelfAttention入参个数
}

uint32_t SelfAttentionAclnnRunner::GetOutputNum() const
{
    return OUTPUT_NUM; // SelfAttention出参个数
}
REG_RUNNER_TYPE(SelfAttentionAclnnRunner);
} // namespace atb