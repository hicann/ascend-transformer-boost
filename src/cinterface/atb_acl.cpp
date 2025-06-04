/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/atb_acl.h"
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/operation/operation_base.h"

#ifdef __cplusplus
extern "C" {
#endif
static atb::VariantPack g_PACK;
const size_t g_MLAINTENSORNUMINT8NOMASK = 9;
const size_t g_MLAINTENSORNUMINT8MASK = 10;
const size_t g_MLAINTENSORNUMNOMASK = 7;
const size_t g_MLAINTENSORNUMMASK = 8;
const size_t g_MLAOUTTENSORNUMCALCRING = 2;
const size_t g_MLAOUTTENSORNUMNOCALCRING = 1;
const size_t g_MLAPPINTENSORNUM = 24;
const size_t g_MLAPPOUTTENSORNUMCACHEMODE = 4;
const size_t g_MLAPPOUTTENSORNUM = 2;
const size_t g_FUSEDADDTOPK_INTENSOR_NUM = 2;
const size_t g_FUSEDADDTOPK_OUTTENSOR_NUM = 2;
const size_t g_RINGMLA_INTENSOR_NUM = 7;
const size_t g_RINGMLA_OUTTENSOR_NUM = 2;
int64_t GetTensorSize(const aclTensor *input)
{
    const op::Shape shape = input->GetViewShape();
    const size_t dims = shape.GetDimNum();
    int64_t size = 1;
    for (size_t i = 0; i < dims; ++i) {
        size *= shape.GetDim(i);
    }
    return size;
}

atb::Status aclTensorToAtbTensor(const aclTensor *aclTensorSrc, atb::Tensor *atbTensorDst)
{
    if (aclTensorSrc == nullptr) {
        atbTensorDst->hostData = nullptr;
        atbTensorDst->deviceData = nullptr;
        return atb::NO_ERROR;
    }
    int64_t *dims = nullptr;
    uint64_t dimCount;
    aclDataType dataType;
    aclFormat format;
    auto status = aclGetViewShape(aclTensorSrc, &dims, &dimCount);
    ATB_CHECK(status == ACL_ERROR_NONE, "aclGetViewShape failed!", return atb::ERROR_INVALID_TENSOR_DIM);
    status = aclGetDataType(aclTensorSrc, &dataType);
    ATB_CHECK(status == ACL_ERROR_NONE, "aclGetDataType failed!", return atb::ERROR_INVALID_TENSOR_DTYPE);
    status = aclGetFormat(aclTensorSrc, &format);
    ATB_CHECK(status == ACL_ERROR_NONE, "aclGetFormat failed!", return atb::ERROR_INVALID_TENSOR_FORMAT);
    atb::TensorDesc desc;
    desc.shape.dimNum = dimCount;
    for (size_t i = 0; i < dimCount; i++) {
        desc.shape.dims[i] = (static_cast<int64_t *>(dims))[i];
    }
    desc.format = format;
    desc.dtype = dataType;
    atbTensorDst->desc = desc;
    atbTensorDst->deviceData = aclTensorSrc->GetData();
    atbTensorDst->hostData = nullptr;
    atbTensorDst->dataSize = GetTensorSize(aclTensorSrc);
    return atb::NO_ERROR;
}

atb::Status aclTensorToAtbTensorHost(const aclTensor *aclTensorSrc, atb::Tensor *atbTensorDst)
{
    if (aclTensorSrc == nullptr) {
        atbTensorDst->hostData = nullptr;
        atbTensorDst->deviceData = nullptr;
        return atb::NO_ERROR;
    }
    int64_t *dims = nullptr;
    uint64_t dimCount;
    aclDataType dataType;
    aclFormat format;
    auto status = aclGetViewShape(aclTensorSrc, &dims, &dimCount);
    ATB_CHECK(status == ACL_ERROR_NONE, "aclGetViewShape failed!", return atb::ERROR_INVALID_TENSOR_DIM);
    status = aclGetDataType(aclTensorSrc, &dataType);
    ATB_CHECK(status == ACL_ERROR_NONE, "aclGetDataType failed!", return atb::ERROR_INVALID_TENSOR_DTYPE);
    status = aclGetFormat(aclTensorSrc, &format);
    ATB_CHECK(status == ACL_ERROR_NONE, "aclGetFormat failed!", return atb::ERROR_INVALID_TENSOR_FORMAT);
    atb::TensorDesc desc;
    desc.shape.dimNum = dimCount;
    for (size_t i = 0; i < dimCount; i++) {
        desc.shape.dims[i] = (static_cast<int64_t *>(dims))[i];
    }
    desc.format = format;
    desc.dtype = dataType;
    atbTensorDst->desc = desc;
    atbTensorDst->deviceData = nullptr;
    atbTensorDst->hostData = aclTensorSrc->GetData();
    atbTensorDst->dataSize = GetTensorSize(aclTensorSrc);
    return atb::NO_ERROR;
}

atb::Status AtbMLAGetWorkspaceSize(const aclTensor *qNope, const aclTensor *qRope, const aclTensor *ctKV,
                                   const aclTensor *kRope, const aclTensor *blockTables, const aclTensor *contextLens,
                                   const aclTensor *mask, const aclTensor *qSeqLen, const aclTensor *qkDescale,
                                   const aclTensor *pvDescale, int32_t headNum, float qkScale, int32_t kvHeadNum,
                                   int maskType, int calcType, uint8_t cacheMode, aclTensor *attenOut, aclTensor *ise,
                                   uint64_t *workspaceSize, atb::Operation **op, atb::Context *context)
{
    atb::infer::MultiLatentAttentionParam param;
    param.headNum = headNum;
    param.qkScale = qkScale;
    param.kvHeadNum = kvHeadNum;
    param.maskType = atb::infer::MultiLatentAttentionParam::MaskType(maskType);
    param.calcType = atb::infer::MultiLatentAttentionParam::CalcType(calcType);
    param.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode(cacheMode);
    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            return st;
        }
    }
    atb::VariantPack pack;
    size_t i = 0;
    size_t counter = 0;
    if (param.cacheMode == atb::infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        if (param.maskType == atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
            pack.inTensors.resize(g_MLAINTENSORNUMINT8NOMASK);
            counter = g_MLAINTENSORNUMINT8NOMASK;
        } else {
            pack.inTensors.resize(g_MLAINTENSORNUMINT8MASK);
            counter = g_MLAINTENSORNUMINT8MASK;
        }
    } else {
        if (param.maskType == atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
            pack.inTensors.resize(g_MLAINTENSORNUMNOMASK);
            counter = g_MLAINTENSORNUMNOMASK;
        } else {
            pack.inTensors.resize(g_MLAINTENSORNUMMASK);
            counter = g_MLAINTENSORNUMMASK;
        }
    }
    if (param.calcType != atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC) {
        pack.inTensors.resize(counter - 1);
    }
    auto status = aclTensorToAtbTensor(qNope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qNope create failed!", return status);
    status = aclTensorToAtbTensor(qRope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qRope create failed!", return status);
    status = aclTensorToAtbTensor(ctKV, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "ctKV create failed!", return status);
    status = aclTensorToAtbTensor(kRope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "kRope create failed!", return status);
    status = aclTensorToAtbTensor(blockTables, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "blockTables create failed!", return status);
    status = aclTensorToAtbTensorHost(contextLens, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "contextLens create failed!", return status);

    if (param.maskType != atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        status = aclTensorToAtbTensor(mask, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "mask create failed!", return status);
    }
    if (param.calcType == atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC) {
        status = aclTensorToAtbTensorHost(qSeqLen, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qSeqLen create failed!", return status);
    }
    if (param.cacheMode == atb::infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        status = aclTensorToAtbTensor(qkDescale, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qkDescale create failed!", return status);
        status = aclTensorToAtbTensor(pvDescale, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "pvDescale create failed!", return status);
    }
    i = 0;
    if (param.calcType != atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING) {
        pack.outTensors.resize(g_MLAOUTTENSORNUMNOCALCRING);
        status = aclTensorToAtbTensor(attenOut, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "attenOut create failed!", return status);
    } else {
        pack.outTensors.resize(g_MLAOUTTENSORNUMCALCRING);
        status = aclTensorToAtbTensor(attenOut, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "calc_type_ring attenOut create failed!", return status);
        status = aclTensorToAtbTensor(ise, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "calc_type_ring ise create failed!", return status);
    }
    atb::Status st = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLA Setup failed!", return st);
    return atb::NO_ERROR;
}

atb::Status AtbMLA(void *workSpcace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context)
{
    atb::Status st = op->Execute(g_PACK, (uint8_t *)(workSpcace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLA Execute failed!", return st);
    return st;
}

atb::Status AtbMLAPreFillGetWorkspaceSize(const aclTensor *q, const aclTensor *qRope, const aclTensor *k,
    const aclTensor *kRope, const aclTensor *v, const aclTensor *qSeqLen, const aclTensor *kvSeqLen,
    const aclTensor *mask, int32_t headNum, float qkScale, int32_t kvHeadNum,
    int maskType, uint8_t cacheMode, aclTensor *attenOut,
    uint64_t *workspaceSize, atb::Operation **op, atb::Context *context)
{
    atb::infer::MultiLatentAttentionParam param;
    param.headNum = headNum;
    param.qkScale = qkScale;
    param.kvHeadNum = kvHeadNum;
    param.maskType = atb::infer::MultiLatentAttentionParam::MaskType(maskType);
    param.calcType = atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL;
    param.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode(cacheMode);
    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            return st;
        }
    }
    atb::VariantPack pack;
    size_t i = 0;

    if (param.maskType == atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        pack.inTensors.resize(g_MLAINTENSORNUMNOMASK);
    } else {
        pack.inTensors.resize(g_MLAINTENSORNUMMASK);
    }

    auto status = aclTensorToAtbTensor(q, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qNope create failed!", return status);
    status = aclTensorToAtbTensor(qRope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qRope create failed!", return status);
    status = aclTensorToAtbTensor(k, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "key create failed!", return status);
    status = aclTensorToAtbTensor(kRope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "kRope create failed!", return status);
    status = aclTensorToAtbTensor(v, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "value create failed!", return status);
    status = aclTensorToAtbTensorHost(qSeqLen, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qSeqLen create failed!", return status);
    status = aclTensorToAtbTensorHost(kvSeqLen, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "kvSeqLen create failed!", return status);

    if (param.maskType == atb::infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
        status = aclTensorToAtbTensor(mask, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "mask create failed!", return status);
    }
  
    pack.outTensors.resize(g_MLAOUTTENSORNUMNOCALCRING);
    status = aclTensorToAtbTensor(attenOut, &(pack.outTensors[0]));
    ATB_CHECK(status == atb::NO_ERROR, "attenOut create failed!", return status);

    atb::Status st = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLA Setup failed!", return st);
    return atb::NO_ERROR;
}

atb::Status AtbMLAPreFill(void* workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context)
{
    atb::Status st = op->Execute(g_PACK, (uint8_t*)(workspace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLA Execute failed!", return st);
    return st;
}

atb::Status AtbMLAPreprocessGetWorkspaceSize(
    const aclTensor *input, const aclTensor *gamma0, const aclTensor *beta0, const aclTensor *quantScale0,
    const aclTensor *quantOffset0, const aclTensor *wdqkv, const aclTensor *deScale0, const aclTensor *bias0,
    const aclTensor *gamma1, const aclTensor *beta1, const aclTensor *quantScale1, const aclTensor *quantOffset1,
    const aclTensor *wuq, const aclTensor *deScale1, const aclTensor *bias1, const aclTensor *gamma2,
    const aclTensor *cos, const aclTensor *sin, const aclTensor *wuk, const aclTensor *kvCache,
    const aclTensor *kvCacheRope, const aclTensor *slotmapping, const aclTensor *ctkvScale, const aclTensor *qNopeScale,
    uint32_t wdqDim, uint32_t qRopeDim, uint32_t kRopeDim, float epsilon, uint32_t qRotaryCoeff, uint32_t kRotaryCoeff,
    bool transposeWdq, bool transposeWuq, bool transposeWuk, uint8_t cacheMode, uint16_t quantMode, aclTensor *qOut0,
    aclTensor *kvCacheOut0, aclTensor *qOut1, aclTensor *kvCacheOut1, uint64_t *workSpaceSize, atb::Operation **op,
    atb::Context *context)
{
    atb::infer::MlaPreprocessParam param;
    param.wdqDim = wdqDim;
    param.qRopeDim = qRopeDim;
    param.kRopeDim = kRopeDim;
    param.epsilon = epsilon;
    param.qRotaryCoeff = qRotaryCoeff;
    param.kRotaryCoeff = kRotaryCoeff;
    param.transposeWdq = transposeWdq;
    param.transposeWuq = transposeWuq;
    param.transposeWuk = transposeWuk;
    param.cacheMode = atb::infer::MlaPreprocessParam::CacheMode(cacheMode);
    param.quantMode = atb::infer::MlaPreprocessParam::QuantMode(quantMode);

    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            return st;
        }
    }
    atb::VariantPack pack;
    size_t i = 0;
    pack.inTensors.resize(g_MLAPPINTENSORNUM);
    auto status = aclTensorToAtbTensor(input, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "input create failed!", return status);
    status = aclTensorToAtbTensor(gamma0, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "gamma0 create failed!", return status);
    status = aclTensorToAtbTensor(beta0, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "beta0 create failed!", return status);
    if (param.quantMode == atb::infer::MlaPreprocessParam::QuantMode::PER_TENSOR_QUANT_ASYMM) {
        status = aclTensorToAtbTensor(quantScale0, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "quantScale0 create failed!", return status);
        status = aclTensorToAtbTensor(quantOffset0, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "quantOffset0 create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(wdqkv, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "wdqkv create failed!", return status);
    status = aclTensorToAtbTensor(deScale0, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "deScale0 create failed!", return status);
    if (param.quantMode != atb::infer::MlaPreprocessParam::QuantMode::PER_TOKEN_QUANT_SYMM &&
        param.quantMode != atb::infer::MlaPreprocessParam::QuantMode::UNQUANT) {
        status = aclTensorToAtbTensor(bias0, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "bias0 create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(gamma1, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "gamma1 create failed!", return status);
    status = aclTensorToAtbTensor(beta1, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "beta1 create failed!", return status);

    if (param.quantMode == atb::infer::MlaPreprocessParam::QuantMode::PER_TENSOR_QUANT_ASYMM) {
        status = aclTensorToAtbTensor(quantScale1, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "quantScale1 create failed!", return status);
        status = aclTensorToAtbTensor(quantOffset1, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "quantOffset1 create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(wuq, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "wuq create failed!", return status);
    status = aclTensorToAtbTensor(deScale1, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "deScale1 create failed!", return status);
    if (param.quantMode != atb::infer::MlaPreprocessParam::QuantMode::PER_TOKEN_QUANT_SYMM &&
        param.quantMode != atb::infer::MlaPreprocessParam::QuantMode::UNQUANT) {
        status = aclTensorToAtbTensor(bias1, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "bias1 create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(gamma2, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "gamma2 create failed!", return status);

    status = aclTensorToAtbTensor(cos, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "cos create failed!", return status);

    status = aclTensorToAtbTensor(sin, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "sin create failed!", return status);

    status = aclTensorToAtbTensor(wuk, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "wuk create failed!", return status);

    status = aclTensorToAtbTensor(kvCache, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "kvCache create failed!", return status);

    if (param.cacheMode != atb::infer::MlaPreprocessParam::CacheMode::KVCACHE) {
        status = aclTensorToAtbTensor(kvCacheRope, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "kvCacheRope create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(slotmapping, &(pack.inTensors[i++]));
    if (param.cacheMode == atb::infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE) {
        status = aclTensorToAtbTensor(ctkvScale, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "ctkvScale create failed!", return status);
        status = aclTensorToAtbTensor(qNopeScale, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qNopeScale create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }

    i = 0;
    if (param.cacheMode != atb::infer::MlaPreprocessParam::CacheMode::KVCACHE) {
        pack.outTensors.resize(g_MLAPPOUTTENSORNUMCACHEMODE);
        status = aclTensorToAtbTensor(qOut0, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qOut0 create failed!", return status);
        status = aclTensorToAtbTensor(kvCacheOut0, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "kvCacheOut0 create failed!", return status);
        status = aclTensorToAtbTensor(qOut1, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qOut1 create failed!", return status);
        status = aclTensorToAtbTensor(kvCacheOut1, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "kvCacheOut1 create failed!", return status);
    } else {
        pack.outTensors.resize(g_MLAPPOUTTENSORNUM);
        status = aclTensorToAtbTensor(qOut0, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qOut0 create failed!", return status);
        status = aclTensorToAtbTensor(kvCacheOut0, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "kvCacheOut0 create failed!", return status);
    }
    atb::Status st = (*op)->Setup(pack, *workSpaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLAPreprocess Setup failed!", return st);
    return atb::NO_ERROR;
}

atb::Status AtbMLAPreprocess(void *workSpace, uint64_t workSpaceSize, atb::Operation *op, atb::Context *context)
{
    atb::Status st = op->Execute(g_PACK, (uint8_t *)(workSpace), workSpaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLAPreprocess Execute failed!", return st);
    return st;
}

atb::Status AtbFusedAddTopkDivGetWorkspaceSize(const aclTensor *x, const aclTensor *addNum, const aclTensor *mappingNum,
                                               const aclTensor *mappingTable, uint32_t groupNum, uint32_t groupTopk,
                                               uint32_t n, uint32_t k, int activationType, bool isNorm, float scale,
                                               bool enableExpertMapping, aclTensor *y, aclTensor *indices,
                                               uint64_t *workSpaceSize, atb::Operation **op, atb::Context *context)
{
    atb::infer::FusedAddTopkDivParam param;
    param.groupNum = groupNum;
    param.groupTopk = groupTopk;
    param.n = n;
    param.k = k;
    param.isNorm = isNorm;
    param.scale = scale;
    param.enableExpertMapping = enableExpertMapping;
    param.activationType = atb::infer::ActivationType(activationType);

    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            return st;
        }
    }
    atb::VariantPack pack;

    size_t intensorNum = g_FUSEDADDTOPK_INTENSOR_NUM;
    if (enableExpertMapping) {
        intensorNum += 2; // 2: mappingNum, mappingTable
    }
    pack.inTensors.resize(intensorNum);
    size_t index = 0;
    auto status = aclTensorToAtbTensor(x, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "x create failed!", return status);
    status = aclTensorToAtbTensor(addNum, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "addNum create failed!", return status);
    if (enableExpertMapping) {
        status = aclTensorToAtbTensor(mappingNum, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "mappingNum create failed!", return status);
        status = aclTensorToAtbTensor(mappingTable, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "mappingTable create failed!", return status);
    }

    index = 0;
    pack.outTensors.resize(g_FUSEDADDTOPK_OUTTENSOR_NUM);
    status = aclTensorToAtbTensor(y, &(pack.outTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "y create failed!", return status);
    status = aclTensorToAtbTensor(indices, &(pack.outTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "indices create failed!", return status);
    (*op)->Setup(pack, *workSpaceSize, context);
    return atb::NO_ERROR;
}

atb::Status AtbFusedAddTopkDiv(void *workSpace, uint64_t workSpaceSize, atb::Operation *op, atb::Context *context)
{
    atb::VariantPack pack;
    atb::Status st = op->Execute(pack, (uint8_t *)(workSpace), workSpaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLAPreprocess Execute failed!", return st);
    return st;
}

atb::Status AtbRingMLAGetWorkspaceSize(const aclTensor *querySplit1, const aclTensor *querySplit2,
                                       const aclTensor *keySplit1, const aclTensor *keySplit2, const aclTensor *value,
                                       const aclTensor *mask, const aclTensor *seqLen, const aclTensor *prevOut,
                                       const aclTensor *prevLse, int32_t headNum, int32_t kvHeadNum, float qkScale,
                                       int kernelType, int maskType, int inputLayout, int calcType, aclTensor *output,
                                       aclTensor *softmaxLse, uint64_t *workSpaceSize, atb::Operation **op,
                                       atb::Context *context)
{
    atb::infer::RingMLAParam param;
    param.headNum = headNum;
    param.kvHeadNum = kvHeadNum;
    param.qkScale = qkScale;
    param.kernelType = atb::infer::RingMLAParam::KernelType(kernelType);
    param.maskType = atb::infer::RingMLAParam::MaskType(maskType);
    param.inputLayout = atb::infer::InputLayout(inputLayout);
    param.calcType = atb::infer::RingMLAParam::CalcType(calcType);
    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            return st;
        }
    }
    atb::VariantPack pack;
    size_t index = 0;
    if (param.calcType == atb::infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT) {
        pack.inTensors.resize(g_RINGMLA_INTENSOR_NUM + 2); // 2: prevOut, prevLse
    } else {
        pack.inTensors.resize(g_RINGMLA_INTENSOR_NUM);
    }

    auto status = aclTensorToAtbTensor(querySplit1, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "querySplit1 create failed!", return status);
    status = aclTensorToAtbTensor(querySplit2, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "querySplit2 create failed!", return status);
    status = aclTensorToAtbTensor(keySplit1, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "keySplit1 create failed!", return status);
    status = aclTensorToAtbTensor(keySplit2, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "keySplit2 create failed!", return status);
    status = aclTensorToAtbTensor(value, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "value create failed!", return status);
    status = aclTensorToAtbTensor(mask, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "mask create failed!", return status);
    status = aclTensorToAtbTensorHost(seqLen, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "seqLen create failed!", return status);
    if (param.calcType == atb::infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT) {
        status = aclTensorToAtbTensor(prevOut, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "prevOut create failed!", return status);
        status = aclTensorToAtbTensor(prevLse, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "prevLse create failed!", return status);
    }

    index = 0;
    pack.outTensors.resize(g_RINGMLA_OUTTENSOR_NUM);
    status = aclTensorToAtbTensor(output, &(pack.outTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "output create failed!", return status);
    status = aclTensorToAtbTensor(softmaxLse, &(pack.outTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "softmaxLse create failed!", return status);
    atb::Status st = (*op)->Setup(pack, *workSpaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbRingMLA Setup failed!", return st);
    return atb::NO_ERROR;
}

atb::Status AtbRingMLA(void *workSpcace, uint64_t workSpaceSize, atb::Operation *op, atb::Context *context)
{
    atb::Status st = op->Execute(g_PACK, (uint8_t *)(workSpcace), workSpaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbRingMLA Execute failed!", return st);
    return st;
}
#ifdef __cplusplus
}
#endif
