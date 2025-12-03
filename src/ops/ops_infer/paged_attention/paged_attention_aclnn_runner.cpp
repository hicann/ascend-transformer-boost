/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "paged_attention_aclnn_runner.h"
#include <securec.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const int64_t INT_MAX_VALUE = 2147483647;

static const int QUERY_ACLNN_TENSOR_IDX = 0;
static const int KEY_ACLNN_TENSOR_IDX = 1;
static const int VALUE_ACLNN_TENSOR_IDX = 2;
static const int BLOCK_TABLE_ACLNN_TENSOR_IDX = 14;
static const int KEY_ANTIQUANT_SCALE_ACLNN_TENSOR_IDX = 17;
static const int VALUE_ANTIQUANT_SCALE_ACLNN_TENSOR_IDX = 19;
static const int ATTENTION_OUT_ACLNN_TENSOR_IDX = 0;
} // namespace

namespace atb {
AclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc PagedAttentionAclnnRunner::aclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc_ = nullptr;
AclnnFusedInferAttentionScoreV3Func PagedAttentionAclnnRunner::aclnnFusedInferAttentionScoreV3Func_ = nullptr;

PagedAttentionAclnnRunner::PagedAttentionAclnnRunner(const infer::PagedAttentionParam &param)
    : AclnnRunner("PagedAttentionAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionAclnnRunner::PagedAttentionAclnnRunner";

    GetTensorNum();
    InitTensorIndex();
    InitAclnnParam();
}

PagedAttentionAclnnRunner::~PagedAttentionAclnnRunner() {}

Status PagedAttentionAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::LoadAclnnFuncs";
    if (aclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc_ && aclnnFusedInferAttentionScoreV3Func_) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile(
        "aclnnFusedInferAttentionScoreV3GetWorkspaceSize", "aclnnFusedInferAttentionScoreV3",
        aclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc_, aclnnFusedInferAttentionScoreV3Func_);
}

Status PagedAttentionAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: "
                  << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclInTensorList.reserve(aclInTensorListNum_);
    aclnnVariantPack_.aclInTensorList.resize(aclInTensorListNum_);
    Status st = CreateQueryAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateKeyAclnnTensorList();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateValueAclnnTensorList();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateBlockTableAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateActualSeqLengthKvAclIntArray();
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.quantType == infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        st = CreateKeyAntiquantScaleAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateValueAntiquantScaleAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
    }

    aclnnVariantPack_.aclOutTensors.reserve(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensorList.reserve(aclOutTensorListNum_);
    aclnnVariantPack_.aclOutTensorList.resize(aclOutTensorListNum_);
    return CreateAttentionOutAclnnTensor();
}

aclnnStatus PagedAttentionAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionAclnnRunner::SetAclNNWorkspaceExecutor";

    aclTensor *query = aclnnVariantPack_.aclInTensors.at(queryAclTensorIndex_)->tensor;
    aclTensorList *key = aclnnVariantPack_.aclInTensorList.at(keyAclTensorListIndex_);
    aclTensorList *value = aclnnVariantPack_.aclInTensorList.at(valueAclTensorListIndex_);
    aclTensor *blockTable = aclnnVariantPack_.aclInTensors.at(blockTableAclTensorIndex_)->tensor;
    aclTensor *keyAntiquantScale = nullptr;
    aclTensor *valueAntiquantScale = nullptr;
    if (param_.quantType == infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        keyAntiquantScale = aclnnVariantPack_.aclInTensors.at(keyAntiquantScaleAclTensorIndex_)->tensor;
        valueAntiquantScale = aclnnVariantPack_.aclInTensors.at(valueAntiquantScaleAclTensorIndex_)->tensor;
    }
    aclTensor *attentionOut = aclnnVariantPack_.aclOutTensors.at(attentionOutAclTensorIndex_)->tensor;

    std::string inputLayoutStr = aclnnParam_.inputLayoutStr;
    auto inputLayout = std::make_unique<char[]>(inputLayoutStr.length() + 1);
    errno_t err = strcpy_s(inputLayout.get(), inputLayoutStr.length() + 1, inputLayoutStr.c_str());
    if (err != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inputLayout strcpy_s failed";
        return ERROR_INVALID_PARAM;
    }
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();
    aclnnStatus ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSizeFunc_(
        query, key, value, nullptr, nullptr, nullptr, actualSeqLengthsKv_, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, blockTable, nullptr, nullptr, keyAntiquantScale, nullptr, valueAntiquantScale, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, aclnnParam_.numHeads, aclnnParam_.scaleValue,
        aclnnParam_.preTokens, aclnnParam_.nextTokens, inputLayout.get(), aclnnParam_.numKeyValueHeads,
        aclnnParam_.sparseMode, aclnnParam_.innerPrecise, aclnnParam_.blockSize, aclnnParam_.antiquantMode,
        aclnnParam_.softmaxLseFlag, aclnnParam_.keyAntiquantMode, aclnnParam_.valueAntiquantMode, attentionOut, nullptr,
        &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret == ACL_SUCCESS) {
        ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "SetAclNNWorkspaceExecutor failed, ret: " << ret;
    }
    return ret;
}

Status PagedAttentionAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionAclnnRunner::LaunchAclnnKernel";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnFusedInferAttentionScoreV3Func_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (actualSeqLengthsKv_) {
        aclnnStatus ret = aclDestroyIntArray(actualSeqLengthsKv_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "actualSeqLengthsKv aclDestroyIntArray failed";
            return ERROR_INTERNAL_ERROR;
        }
        actualSeqLengthsKv_ = nullptr;
    }
    if (ret == ACL_SUCCESS) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "LaunchAclnnKernel failed, ret: " << ret;
    return ERROR_CANN_ERROR;
}

void PagedAttentionAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = 4; // 4: query, key, value, blockTable
    if (param_.quantType == infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        aclInTensorNum_ += 2; // 2: keyAntiquantScale, valueAntiquantScale
    }
    aclOutTensorNum_ = 1;
    aclInTensorListNum_ = 2; // 2: key, value
    aclOutTensorListNum_ = 0;
    ATB_LOG(INFO) << GetLogPrefix() << "aclInTensorNum: " << aclInTensorNum_ << ", aclOutTensorNum: " << aclOutTensorNum_;
}

void PagedAttentionAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    aclInTensorListIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorIndex_ = 0;
    aclOutTensorListIndex_ = 0;
    queryAclTensorIndex_ = 0;
    keyAclTensorListIndex_ = 0;
    valueAclTensorListIndex_ = 0;
    blockTableAclTensorIndex_ = 0;
    keyAntiquantScaleAclTensorIndex_ = 0;
    valueAntiquantScaleAclTensorIndex_ = 0;
    attentionOutAclTensorIndex_ = 0;
}

void PagedAttentionAclnnRunner::InitAclnnParam()
{
    aclnnParam_.numHeads = param_.headNum;
    aclnnParam_.scaleValue = param_.qkScale;
    aclnnParam_.preTokens = INT_MAX_VALUE;
    aclnnParam_.nextTokens = INT_MAX_VALUE;
    if (param_.inputLayout == infer::TYPE_BSND) {
        aclnnParam_.inputLayoutStr = "BSND";
    } else if (param_.inputLayout == infer::TYPE_BNSD) {
        aclnnParam_.inputLayoutStr = "BNSD";
    }
    aclnnParam_.numKeyValueHeads = param_.kvHeadNum;
}

Status PagedAttentionAclnnRunner::CreateQueryAclnnTensor()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateQueryAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, QUERY_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 4; // 4: [B, S, N, D]
    viewShape.dims[0] = storageShape.dims[0];
    viewShape.dims[1] = 1;
    viewShape.dims[2] = storageShape.dims[1]; // 2: N(head_num)
    viewShape.dims[3] = storageShape.dims[2]; // 3: D(head_size); 2: head_size
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "query aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    queryAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status PagedAttentionAclnnRunner::CreateKeyAclnnTensorList()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateKeyAclnnTensorList";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, KEY_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    aclnnParam_.blockSize = storageShape.dims[1];
    Dims viewShape;
    viewShape.dimNum = 3; // [B, S, H]
    viewShape.dims[0] = storageShape.dims[0];
    viewShape.dims[1] = storageShape.dims[1];
    viewShape.dims[2] = storageShape.dims[2] * storageShape.dims[3]; // 2: H(hidden_size); 2: head_num; 3: head_size
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "key aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    aclTensor *keyTensors[1] = {aclnnTensorPtr->tensor};
    auto keyTensorList = aclCreateTensorList(keyTensors, 1);
    if (keyTensorList) {
        aclnnVariantPack_.aclInTensorList.at(aclInTensorListIndex_) = keyTensorList;
        aclInTensorIndex_++;
        keyAclTensorListIndex_ = aclInTensorListIndex_++;
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "key aclCreateTensorList failed";
    return ERROR_INTERNAL_ERROR;
}

Status PagedAttentionAclnnRunner::CreateValueAclnnTensorList()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateValueAclnnTensorList";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, VALUE_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 3;
    viewShape.dims[0] = storageShape.dims[0];
    viewShape.dims[1] = storageShape.dims[1];
    viewShape.dims[2] = storageShape.dims[2] * storageShape.dims[3]; // 2: H(hidden_size); 2: head_num; 3: head_size
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "value aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    aclTensor *valueTensors[1] = {aclnnTensorPtr->tensor};
    auto valueTensorList = aclCreateTensorList(valueTensors, 1);
    if (valueTensorList) {
        aclnnVariantPack_.aclInTensorList.at(aclInTensorListIndex_) = valueTensorList;
        aclInTensorIndex_++;
        valueAclTensorListIndex_ = aclInTensorListIndex_++;
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "value aclCreateTensorList failed";
    return ERROR_INTERNAL_ERROR;
}

Status PagedAttentionAclnnRunner::CreateBlockTableAclnnTensor()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateBlockTableAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, BLOCK_TABLE_ACLNN_TENSOR_IDX);
    Dims viewShape = atbTensor.desc.shape;
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "blockTable aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    blockTableAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status PagedAttentionAclnnRunner::CreateActualSeqLengthKvAclIntArray()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateActualSeqLengthKvAclIntArray";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    if (atbTensor.hostData == nullptr) {
        ATB_LOG(ERROR) << "contextLen tensor host data is null";
        return ERROR_INVALID_TENSOR_ADDR;
    }
    if (actualSeqLengthsKv_) {
        aclnnStatus ret = aclDestroyIntArray(actualSeqLengthsKv_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "actualSeqLengthsKv aclDestroyIntArray failed";
            return ERROR_INTERNAL_ERROR;
        }
        actualSeqLengthsKv_ = nullptr;
    }
    SVector<int32_t> contextLensInt32;
    uint64_t dataSize = atbTensor.dataSize / 4; // 4: int32 size
    contextLensInt32.reserve(dataSize);
    contextLensInt32.resize(dataSize);
    if (memcpy_s(contextLensInt32.data(), atbTensor.dataSize, atbTensor.hostData, atbTensor.dataSize) != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "contextLens memcpy_s failed";
        return ERROR_INTERNAL_ERROR;
    }
    std::vector<int64_t> contextLensInt64;
    contextLensInt64.reserve(dataSize);
    contextLensInt64.resize(dataSize);
    for (uint64_t i = 0; i < dataSize; i++) {
        contextLensInt64.at(i) = static_cast<int64_t>(contextLensInt32.at(i));
    }
    actualSeqLengthsKv_ = aclCreateIntArray(contextLensInt64.data(), dataSize);
    if (actualSeqLengthsKv_) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << "actualSeqLengthsKv aclCreateIntArray failed!";
    return ERROR_INTERNAL_ERROR;
}

Status PagedAttentionAclnnRunner::CreateKeyAntiquantScaleAclnnTensor()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateKeyAntiquantScaleAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, KEY_ANTIQUANT_SCALE_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 2; // 2: [1, H]
    viewShape.dims[0] = 1;
    viewShape.dims[1] = storageShape.dims[0];
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "keyAntiquantScale aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    keyAntiquantScaleAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status PagedAttentionAclnnRunner::CreateValueAntiquantScaleAclnnTensor()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateValueAntiquantScaleAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, VALUE_ANTIQUANT_SCALE_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 2; // 2: [1, H]
    viewShape.dims[0] = 1;
    viewShape.dims[1] = storageShape.dims[0];
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "valueAntiquantScale aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    valueAntiquantScaleAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status PagedAttentionAclnnRunner::CreateAttentionOutAclnnTensor()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateAttentionOutAclnnTensor";
    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, ATTENTION_OUT_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 4; // 4: [B, S, N, D]
    viewShape.dims[0] = storageShape.dims[0];
    viewShape.dims[1] = 1;
    viewShape.dims[2] = storageShape.dims[1]; // 2: N(head_num)
    viewShape.dims[3] = storageShape.dims[2]; // 3: D(head_size); 2: head_size
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "attentionOut aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    attentionOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

std::shared_ptr<AclNNTensor> PagedAttentionAclnnRunner::InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex)
{
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->tensorIdx = aclnnTensorIndex;
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    return aclnnTensorPtr;
}

REG_RUNNER_TYPE(PagedAttentionAclnnRunner);
} // namespace atb
