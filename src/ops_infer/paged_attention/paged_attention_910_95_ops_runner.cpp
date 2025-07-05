/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "paged_attention_910_95_ops_runner.h"
#include <atbops/params/params.h>
#include <atb/utils/config.h>
#include <atb/utils/singleton.h>
#include "atb/utils/log.h"
#include "paged_attention_operation.h"

namespace atb {

PagedAttention91095OpsRunner::PagedAttention91095OpsRunner(const infer::PagedAttentionParam &param)
    : OpsRunner("PagedAttentionOpsRunner", RUNNER_TYPE_PAGED_ATTENTION), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
    ATB_LOG(INFO) << "PagedAttention91095OpsRunner::PagedAttention91095OpsRunner called";
}

PagedAttention91095OpsRunner::~PagedAttention91095OpsRunner() {}

Status PagedAttention91095OpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    if (param_.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        return SetupKernelGraphDequantFusion();
    } else {
        return SetupKernelGraphBase();
    }
}

Status PagedAttention91095OpsRunner::SetupKernelGraphBase()
{
    ATB_LOG(INFO) << "PagedAttention91095OpsRunner::SetupKernelGraphBase";

    InitKernelGraph(12, 1); // 12: input tensor num

    size_t inTensorIndex = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorIndex++);
    Mki::Tensor &keyCache = kernelGraph_.inTensors.at(inTensorIndex++);
    Mki::Tensor &valueCache = kernelGraph_.inTensors.at(inTensorIndex++);
    Mki::Tensor &blockTables = kernelGraph_.inTensors.at(inTensorIndex++);
    inTensorIndex++;
    Mki::Tensor &mask = nullTensor_;
    Mki::Tensor &kDescale = nullTensor_;
    Mki::Tensor &kOffset = nullTensor_;
    Mki::Tensor &vDescale = nullTensor_;
    Mki::Tensor &vOffset = nullTensor_;
    Mki::Tensor &razorOffset = nullTensor_;
    Mki::Tensor &pScale = nullTensor_;
    Mki::Tensor &logN = nullTensor_;

    Mki::Tensor &output = kernelGraph_.outTensors.at(0);

    auto &pagedAttentionNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::PagedAttention pagedAttentionParam;
    pagedAttentionParam.type = AtbOps::OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND;
    pagedAttentionParam.headSize = param_.headNum;
    pagedAttentionParam.tor = param_.qkScale;
    pagedAttentionParam.kvHead = param_.kvHeadNum;
    pagedAttentionParam.maskType = GetMaskType();
    pagedAttentionParam.scaleType = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN) ?
                                        AtbOps::OpParam::PagedAttention::SCALE_LOGN_FP32 :
                                        AtbOps::OpParam::PagedAttention::SCALE_TOR;
    pagedAttentionParam.compressHead =
        (param_.compressType == infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD);
    pagedAttentionNode.opDesc = {0, "PagedAttentionOperation", pagedAttentionParam};
    pagedAttentionNode.inTensors = {&query,   &keyCache, &valueCache, &blockTables, &mask,   &kDescale,
                                    &kOffset, &vDescale, &vOffset,    &razorOffset, &pScale, &logN};
    pagedAttentionNode.outTensors = {&output};
    SVector<int> inputs = {0, 1, 1, 0, 0, 0, 0};
    pagedAttentionNode.inputLens = inputs;
    pagedAttentionNode.inTensorViewFuncs.resize(pagedAttentionNode.inTensors.size()); // view
    pagedAttentionNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {        // format, dtype 设置
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
    pagedAttentionNode.inTensorViewFuncs.at(0) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), 1, oldDims.at(1), oldDims.at(2)}; // 2: dim2
    };
    pagedAttentionNode.inTensorViewFuncs.at(1) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)}; // 2: dim2; 3: dim3
    };
    pagedAttentionNode.inTensorViewFuncs.at(2) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)}; // 2: dim2; 3: dim3
    };
    newParam_.batchRunStatus.reserve(128); // 128: 预留大小
    newParam_.contextLens.reserve(128);    // 128: 预留大小
    newParam_.qLens.reserve(128);          // 128: 预留大小

    return NO_ERROR;
}

Status PagedAttention91095OpsRunner::SetupKernelGraphDequantFusion()
{
    ATB_LOG(INFO) << "PagedAttention91095OpsRunner::SetupKernelGraphDequantFusion";

    InitKernelGraph(12, 1); // 12: input tensor num

    size_t inTensorIndex = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorIndex++);
    Mki::Tensor &keyCache = kernelGraph_.inTensors.at(inTensorIndex++);
    Mki::Tensor &valueCache = kernelGraph_.inTensors.at(inTensorIndex++);
    Mki::Tensor &blockTables = kernelGraph_.inTensors.at(inTensorIndex++);
    inTensorIndex++;
    Mki::Tensor &mask = nullTensor_;
    Mki::Tensor &kDescale = kernelGraph_.inTensors.at(inTensorIndex++);
    Mki::Tensor &kOffset = nullTensor_;
    Mki::Tensor &vDescale = kernelGraph_.inTensors.at(inTensorIndex++);
    Mki::Tensor &vOffset = nullTensor_;
    Mki::Tensor &razorOffset = nullTensor_;
    Mki::Tensor &pScale = nullTensor_;
    Mki::Tensor &logN = nullTensor_;

    Mki::Tensor &output = kernelGraph_.outTensors.at(0);

    auto &pagedAttentionNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::PagedAttention pagedAttentionParam;
    pagedAttentionParam.type = AtbOps::OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND;
    pagedAttentionParam.headSize = param_.headNum;
    pagedAttentionParam.tor = param_.qkScale;
    pagedAttentionParam.kvHead = param_.kvHeadNum;
    pagedAttentionParam.maskType = GetMaskType();
    pagedAttentionParam.scaleType = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN) ?
                                        AtbOps::OpParam::PagedAttention::SCALE_LOGN_FP32 :
                                        AtbOps::OpParam::PagedAttention::SCALE_TOR;
    pagedAttentionParam.compressHead =
        (param_.compressType == infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD);
    pagedAttentionNode.opDesc = {0, "PagedAttentionOperation", pagedAttentionParam};
    pagedAttentionNode.inTensors = {&query,   &keyCache, &valueCache, &blockTables, &mask,   &kDescale,
                                    &kOffset, &vDescale, &vOffset,    &razorOffset, &pScale, &logN};
    pagedAttentionNode.outTensors = {&output};
    SVector<int> inputs = {0, 1, 1, 0, 0, 0, 0};
    pagedAttentionNode.inputLens = inputs;
    pagedAttentionNode.inTensorViewFuncs.resize(pagedAttentionNode.inTensors.size()); // view
    pagedAttentionNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {     // format, dtype 设置
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
    pagedAttentionNode.inTensorViewFuncs.at(0) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), 1, oldDims.at(1), oldDims.at(2)}; // 2: dim2
    };
    pagedAttentionNode.inTensorViewFuncs.at(1) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)}; // 2: dim2; 3: dim3
    };
    pagedAttentionNode.inTensorViewFuncs.at(2) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)}; // 2: dim2; 3: dim3
    };
    pagedAttentionNode.inTensorViewFuncs.at(5) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
        newDims = {1, oldDims.at(0)};
    };
    pagedAttentionNode.inTensorViewFuncs.at(7) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
        newDims = {1, oldDims.at(0)};
    };
    newParam_.batchRunStatus.reserve(128); // 128: 预留大小
    newParam_.contextLens.reserve(128);    // 128: 预留大小
    newParam_.qLens.reserve(128);          // 128: 预留大小

    return NO_ERROR;
}

AtbOps::OpParam::PagedAttention::MaskType PagedAttention91095OpsRunner::GetMaskType() const
{
    return static_cast<AtbOps::OpParam::PagedAttention::MaskType>(param_.maskType);
}

Status PagedAttention91095OpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    int batchRunStatusIndex = 5; // 5: batchRunStatus 位置
    if (param_.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
        batchRunStatusIndex++;
    }
    int qLensIndex = 5; // 5: batchRunStatus 位置
    if (param_.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
        qLensIndex++;
    }
    if (param_.batchRunStatusEnable) {
        qLensIndex++;
    }
    if (param_.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        qLensIndex += 2; // 2: kv descale
    }
    if (param_.hasQuantOffset) {
        qLensIndex += 2; // 2: kv offset
    }
    bool ret = newParam_.BuildFromTensor91095(opsTensorPack.inTensors);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    auto &pagedAttentionNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::PagedAttention pagedAttentionParam;
    SetPaParam(pagedAttentionParam);
    pagedAttentionNode.opDesc = {0, "PagedAttentionOperation", pagedAttentionParam};
    return NO_ERROR;
}

void PagedAttention91095OpsRunner::InitKernelGraph(size_t inTensorNum, size_t outTensorNum, size_t nodeNum,
                                                   size_t internalTensorNum)
{
    kernelGraph_.inTensors.resize(inTensorNum);
    kernelGraph_.outTensors.resize(outTensorNum);
    kernelGraph_.nodes.resize(nodeNum);
    kernelGraph_.internalTensors.resize(internalTensorNum);
}

void PagedAttention91095OpsRunner::SetPaParam(AtbOps::OpParam::PagedAttention &pagedAttentionParam)
{
    pagedAttentionParam.type = AtbOps::OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND;
    pagedAttentionParam.headSize = param_.headNum;
    pagedAttentionParam.tor = param_.qkScale;
    pagedAttentionParam.kvHead = param_.kvHeadNum;
    pagedAttentionParam.maskType = GetMaskType();
    pagedAttentionParam.scaleType = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN) ?
                                        AtbOps::OpParam::PagedAttention::SCALE_LOGN_FP32 :
                                        AtbOps::OpParam::PagedAttention::SCALE_TOR;
    pagedAttentionParam.kvSeqLen = newParam_.contextLens;
    pagedAttentionParam.batchRunStatus = newParam_.batchRunStatus;
    pagedAttentionParam.qSeqLen = newParam_.qLens;
    pagedAttentionParam.dataShapeType = static_cast<AtbOps::OpParam::PagedAttention::DataShapeType>(param_.inputLayout);
    pagedAttentionParam.quantType = static_cast<AtbOps::OpParam::PagedAttention::QuantType>(param_.quantType);
    pagedAttentionParam.outDataType = static_cast<Mki::TensorDType>(param_.outDataType);
    pagedAttentionParam.compressHead =
        (param_.compressType != infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED);
    if (param_.quantType == infer::PagedAttentionParam::TYPE_DEQUANT_FUSION) {
        pagedAttentionParam.identityM.resize(32 * 32); // 32: 单位阵大小
        for (size_t i = 0; i < 32; ++i) {              // 32: 单位阵大小
            pagedAttentionParam.identityM[i * 32 + i] = 1;
        }
    }
}
} // namespace atb