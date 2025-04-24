/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "multi_latent_attention_operation.h"
#include "multi_latent_attention_ops_runner.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/core/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 6;
static const uint32_t OUT_TENSOR_NUM_1 = 1;
static const uint32_t OUT_TENSOR_NUM_2 = 2;
static const uint32_t KVCACHE_INDEX = 2;
static const uint32_t KVCACHE_ROPE_INDEX = 3;
static const uint32_t BLOCK_TABLES_INDEX = 4;
static const uint32_t CONTEXTLENS_INDEX = 5;
static const uint32_t INNER_DIM_512 = 512;
static const uint32_t INNER_DIM_64 = 64;
static const uint32_t INNER_DIM_32 = 32;
static const uint32_t INNER_DIM_16 = 16;
static const uint32_t INNER_DIM_4 = 4;
static const uint32_t NZ_ALIGN_32 = 32;
static const uint32_t NZ_ALIGN_16 = 16;
static const uint32_t MAX_BATCH_SIZE_8192 = 8192;
} // namespace

namespace atb {

static bool ParamRangeCheck(const infer::MultiLatentAttentionParam &opParam);

template <> Status CreateOperation(const infer::MultiLatentAttentionParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "only support Atlas 800I A2/A3 Inference Product";
        return ERROR_INVALID_PARAM;
    }
    if (!ParamRangeCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (opParam.cacheMode == infer::MultiLatentAttentionParam::CacheMode::KVCACHE) {
        ATB_LOG(ERROR) << "dont support cacheMode KVCACHE yet";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.calcType != infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC &&
        opParam.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        ATB_LOG(ERROR) << "only mtp(CALC_TYPE_SPEC) support mask";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.calcType != infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_UNDEFINED &&
        opParam.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        ATB_LOG(ERROR) << "mtp(CALC_TYPE_SPEC) and ring dont support quant";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING &&
        opParam.cacheMode != infer::MultiLatentAttentionParam::CacheMode::KROPE_CTKV) {
        ATB_LOG(ERROR) << "mtp(CALC_TYPE_RING) only support krppe ctkv";
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new (std::nothrow) MultiLatentAttentionOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new MultiLatentAttentionOperation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

static bool ParamRangeCheck(const infer::MultiLatentAttentionParam &opParam)
{
    if (opParam.headNum != 8 && opParam.headNum != 16 && opParam.headNum != 32 && // 8, 16, 32: headNum
        opParam.headNum != 64 && opParam.headNum != 128) {                        // 64, 128: headNum
        ATB_LOG(ERROR) << "headNum should be {8,16,32,64,128}";
        return false;
    }
    if (opParam.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE &&
        opParam.headNum == 128) { // 128: headNum
        ATB_LOG(ERROR) << "headNum should not be 128 with INT8_NZCACHE";
        return false;
    }
    if (opParam.qkScale <= 0 || opParam.qkScale > 1) {
        ATB_LOG(ERROR) << "qkScale should > 0 and <= 1";
        return false;
    }
    if (opParam.kvHeadNum != 1) {
        ATB_LOG(ERROR) << "kvHeadNum should be 1, only support MQA";
        return false;
    }
    if (opParam.maskType < infer::MultiLatentAttentionParam::MaskType::UNDEFINED ||
        opParam.maskType > infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
        ATB_LOG(ERROR) << "invalid maskType";
        return false;
    }
    if (opParam.calcType < infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_UNDEFINED ||
        opParam.calcType > infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING) {
        ATB_LOG(ERROR) << "invalid calcType";
        return false;
    }
    if (opParam.cacheMode > infer::MultiLatentAttentionParam::CacheMode::NZCACHE) {
        ATB_LOG(ERROR) << "invalid cacheMode";
        return false;
    }
    return true;
}

MultiLatentAttentionOperation::MultiLatentAttentionOperation(const infer::MultiLatentAttentionParam &param)
    : OperationBase("MultiLatentAttentionOperation"), param_(param)
{
    std::string opIrKeyStr;
    opIrKeyStr += "MultiLatentAttentionOperation";
    if (param_.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        opIrKeyStr += "Mask";
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC) {
        opIrKeyStr += "Qlens";
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING) {
        opIrKeyStr += "IsRing";
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        opIrKeyStr += "Int8Nz";
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::NZCACHE) {
        opIrKeyStr += "Nz";
    }
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKeyStr);
}

MultiLatentAttentionOperation::~MultiLatentAttentionOperation() {}

uint32_t MultiLatentAttentionOperation::GetInputNum() const
{
    uint32_t intensorNumBase = IN_TENSOR_NUM;
    if (param_.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        intensorNumBase++;
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC) {
        intensorNumBase++;
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        intensorNumBase += 2; // 2: qDescale kDescale
    }
    return intensorNumBase;
}

uint32_t MultiLatentAttentionOperation::GetOutputNum() const
{
    return param_.calcType != infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING ? OUT_TENSOR_NUM_1 :
                                                                                           OUT_TENSOR_NUM_2;
}

Status MultiLatentAttentionOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                     SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).dtype = inTensorDescs.at(1).dtype;
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING) {
        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(1).shape.dims[2] = 1; // 2: dim2
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                     const SVector<Tensor> &outTensors) const
{
    (void)outTensors;
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::QKVDimCheck(const SVector<TensorDesc> &inTensorDesc) const
{
    int64_t numTokens = inTensorDesc.at(0).shape.dims[0];
    int64_t numBlocks = inTensorDesc.at(KVCACHE_INDEX).shape.dims[0];
    int64_t blockSize = inTensorDesc.at(KVCACHE_INDEX).shape.dims[1];
    if (blockSize > 128) { // 128 : maxblocksize
        ATB_LOG(ERROR) << "blockSize shoule <= 128";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[0] != numTokens) {
        ATB_LOG(ERROR) << "dim 0 of query(intensor0) and queryRope(intensor1) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[0] != numBlocks ||
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[1] != blockSize) {
        ATB_LOG(ERROR) << "dim 0 and dim 1 of kvCache(intensor2) and kvCacheRope(intensor3) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_INDEX).shape.dims[2] != param_.kvHeadNum ||      // 2: dim 2
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[2] != param_.kvHeadNum) { // 2: dim 2
        ATB_LOG(ERROR) << "dim 1 of kvCache(intensor2) and kvCacheRope(intensor3) equal to kvHeadNum";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[1] != param_.headNum || inTensorDesc.at(1).shape.dims[1] != param_.headNum) {
        ATB_LOG(ERROR) << "dim 1 of query(intensor0) and queryRope(intensor1) equal to headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[2] != INNER_DIM_512 ||             // 2: dim 2
        inTensorDesc.at(KVCACHE_INDEX).shape.dims[3] != INNER_DIM_512) { // 3: dim 3
        ATB_LOG(ERROR) << "head_size of query(intensor0) and kvCache(intensor2) should be 512";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[2] != INNER_DIM_64 ||                  // 2: dim 2
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[3] != INNER_DIM_64) { // 3: dim 3
        ATB_LOG(ERROR) << "head_size of queryRope(intensor1) and kvCacheRope(intensor3) should be 64";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::QKVDimCheckNz(const SVector<TensorDesc> &inTensorDesc) const
{
    int64_t numTokens = inTensorDesc.at(0).shape.dims[0];
    int64_t numBlocks = inTensorDesc.at(KVCACHE_INDEX).shape.dims[0];
    int64_t blockSize = inTensorDesc.at(KVCACHE_INDEX).shape.dims[2]; // 2: dim 2
    if (blockSize > 128) {                                            // 128 : maxblocksize
        ATB_LOG(ERROR) << "blockSize shoule <= 128";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[0] != numTokens) {
        ATB_LOG(ERROR) << "dim 0 of query(intensor0) and queryRope(intensor1) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[0] != numBlocks ||
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[2] != blockSize) { // 2: dim 2
        ATB_LOG(ERROR) << "dim 0 and dim 2 of kvCache(intensor2) and kvCacheRope(intensor3) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_INDEX).shape.dims[3] != NZ_ALIGN_16 ||      // 3: dim 3
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[3] != NZ_ALIGN_16) { // 3: dim 3
        ATB_LOG(ERROR) << "dim 3 of kvCache(intensor2) and kvCacheRope(intensor3) should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[1] != param_.headNum || inTensorDesc.at(1).shape.dims[1] != param_.headNum) {
        ATB_LOG(ERROR) << "dim 1 of query(intensor0) and queryRope(intensor1) equal to headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[2] != INNER_DIM_512 ||            // 2: dim 2
        inTensorDesc.at(KVCACHE_INDEX).shape.dims[1] != INNER_DIM_32) { // 1: dim 1
        ATB_LOG(ERROR) << "head_size of query(intensor0) should be 512, dim 1 of kvCache(intensor2) should be 32";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[2] != INNER_DIM_64 ||                 // 2: dim 2
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[1] != INNER_DIM_4) { // 1: dim 1
        ATB_LOG(ERROR) << "head_size of queryRope(intensor1) should be 64, dim 1 of kvCacheRope(intensor3) should be 4";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::QKVDimCheckInt8Nz(const SVector<TensorDesc> &inTensorDesc) const
{
    int64_t numTokens = inTensorDesc.at(0).shape.dims[0];
    int64_t numBlocks = inTensorDesc.at(KVCACHE_INDEX).shape.dims[0];
    int64_t blockSize = inTensorDesc.at(KVCACHE_INDEX).shape.dims[2]; // 2: dim 2
    if (blockSize > 128) {                                            // 128 : maxblocksize
        ATB_LOG(ERROR) << "blockSize shoule <= 128";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[0] != numTokens) {
        ATB_LOG(ERROR) << "dim 0 of query(intensor0) and queryRope(intensor1) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[0] != numBlocks ||
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[2] != blockSize) { // 2: dim 2
        ATB_LOG(ERROR) << "dim 0 and dim 2 of kvCache(intensor2) and kvCacheRope(intensor3) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_INDEX).shape.dims[3] != NZ_ALIGN_32 ||      // 3: dim 3
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[3] != NZ_ALIGN_16) { // 3: dim 3
        ATB_LOG(ERROR) << "dim 3 of kvCache(intensor2) should be 32 and kvCacheRope(intensor3) should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[1] != param_.headNum || inTensorDesc.at(1).shape.dims[1] != param_.headNum) {
        ATB_LOG(ERROR) << "dim 1 of query(intensor0) and queryRope(intensor1) equal to headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[2] != INNER_DIM_512 || // 2: dim 2
        inTensorDesc.at(KVCACHE_INDEX).shape.dims[1] != INNER_DIM_16) {
        ATB_LOG(ERROR) << "head_size of query(intensor0) should be 512, dim 1 of kvCache(intensor2) should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[2] != INNER_DIM_64 || // 2: dim 2
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[1] != INNER_DIM_4) {
        ATB_LOG(ERROR) << "head_size of queryRope(intensor1) should be 64, dim 1 of kvCacheRope(intensor3) should be 4";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::DimCheck(const SVector<TensorDesc> &inTensorDesc) const
{
    if (inTensorDesc.at(0).shape.dimNum != 3 ||                  // 0: query 3: 3 dims
        inTensorDesc.at(1).shape.dimNum != 3 ||                  // 1: queryRope 3: 3 dims
        inTensorDesc.at(KVCACHE_INDEX).shape.dimNum != 4 ||      // 4: 4 dims
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dimNum != 4 || // 4: 4 dims
        inTensorDesc.at(BLOCK_TABLES_INDEX).shape.dimNum != 2 || // 2: 2 dims
        inTensorDesc.at(CONTEXTLENS_INDEX).shape.dimNum != 1) {  // 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t numTokens = inTensorDesc.at(0).shape.dims[0];
    int64_t batch = inTensorDesc.at(BLOCK_TABLES_INDEX).shape.dims[0];
    if (batch > MAX_BATCH_SIZE_8192) {
        ATB_LOG(ERROR) << "batch should <= " << MAX_BATCH_SIZE_8192;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_UNDEFINED && numTokens != batch) {
        ATB_LOG(ERROR) << "numTokens and batch should be same in decoder stage";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(CONTEXTLENS_INDEX).shape.dims[0] != batch) {
        ATB_LOG(ERROR) << "dim 0 of block_tables(intensor4) and contextLens(intensor5) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    Status st = NO_ERROR;
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::KROPE_CTKV) {
        st = QKVDimCheck(inTensorDesc);
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        st = QKVDimCheckInt8Nz(inTensorDesc);
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::NZCACHE) {
        st = QKVDimCheckNz(inTensorDesc);
    }
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> MultiLatentAttentionOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<MultiLatentAttentionOpsRunner>(param_);
}

nlohmann::json MultiLatentAttentionOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb