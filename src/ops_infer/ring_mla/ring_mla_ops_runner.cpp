/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ring_mla_ops_runner.h"
#include <cmath>
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "param.h"

namespace {
// query, key, value, mask, seqLen, prevOut (optional), prevLse(optional)
static constexpr uint32_t VALUE_TENSOR_POS = 2;
static constexpr uint32_t SEQLEN_TENSOR_POS = 4;
static constexpr uint32_t IN_TENSOR_NUM = 14;
static constexpr uint32_t OUT_TENSOR_NUM = 2;
} // namespace


namespace atb {

RingMLAOpsRunner::RingMLAOpsRunner(const infer::RingMLAParam &param)
    : OpsRunner("RingMLAOpsRunner", RUNNER_TYPE_RING_MLA), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
    isInputSoftmaxLse_ = param_.calcType == infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT;
    isNoMask_ = param_.maskType == infer::RingMLAParam::MaskType::NO_MASK;
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM);
    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);

    int inTensorStart = 0;
    Mki::Tensor *query = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *key = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *value = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *layerId = &nullTensor_;
    Mki::Tensor *mask = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *seqLen = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *prevOut = isInputSoftmaxLse_ ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *prevLse = isInputSoftmaxLse_ ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    if (isNoMask_) {
        mask = &nullTensor_;
    }

    int outTensorStart = 0;
    Mki::Tensor *attnOut = &kernelGraph_.outTensors.at(outTensorStart++);
    Mki::Tensor *softmaxLse = &kernelGraph_.outTensors.at(outTensorStart++);

    ATB_LOG(INFO) << GetLogPrefix() << "Ctor seqLen dataSize:" << seqLen->dataSize;
    kernelGraph_.nodes.resize(1);
    auto &RingMLANode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::RINGMLA RingMLAParam;
    SetRingMLAParam(RingMLAParam);

    RingMLANode.opDesc = {0, "RingAttentionOperation", RingMLAParam};

    // flashAttentionEncoderNode.inTensors = {&query, &key, value, layerId, mask, slopes, qkDescale, qkOffset,
    // vpvDescale, vpvOffset, pScale, logN};
    RingMLANode.inTensors = {query,        key,          value,        layerId,      mask,
                             &nullTensor_, &nullTensor_, &nullTensor_, &nullTensor_, &nullTensor_,
                             &nullTensor_, &nullTensor_, prevOut,      prevLse};
    RingMLANode.outTensors = {attnOut, softmaxLse};
    RingMLANode.inTensorViewFuncs.resize(RingMLANode.inTensors.size()); // view
    RingMLANode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) { // format, dtype 设置
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
}

Status RingMLAOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    // query, key, value, mask, seqlen,
    RingMLAVariantPackParam newParam;
    bool ret = newParam.BuildFromTensor(opsTensorPack.inTensors, SEQLEN_TENSOR_POS);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    auto &RingMLANode = kernelGraph_.nodes.at(0); // 0: RingMLA节点位置

    AtbOps::OpParam::RINGMLA RingMLAParam;
    SetRingMLAParam(RingMLAParam);
    RingMLAParam.qSeqLen = newParam.qSeqLen;
    RingMLAParam.kvSeqLen = newParam.kvSeqLen;
    if (RingMLAParam.kvSeqLen.size() == 0) {
        RingMLAParam.kvSeqLen = newParam.qSeqLen;
    }

    uint32_t hiddenSizePos = kernelGraph_.inTensors.at(VALUE_TENSOR_POS).desc.dims.size() - 1;
    int32_t hiddenSizeV = static_cast<int32_t>(kernelGraph_.inTensors.at(VALUE_TENSOR_POS).desc.dims[hiddenSizePos]);
    // if (hiddenSizePos == 1) { // [batch * seqLen, HiddenSize]
    //     int32_t headNumV = (RingMLAParam.kvHead > 0) ? RingMLAParam.kvHead : RingMLAParam.headSize;
    //     // RingMLAParam.headSize = param_.headNum, 一定大于0
    //     RingMLAParam.headDimV = hiddenSizeV / headNumV;
    // } else { // [batch, seqLen, headNum, headSize]
    //     RingMLAParam.headDimV = hiddenSizeV;
    // }

    ATB_LOG(INFO) << GetLogPrefix() << "seqLen dataSize: " << newParam.qSeqLen.size();
    ATB_LOG(INFO) << GetLogPrefix() << "kvSeqLen dataSize: " << newParam.kvSeqLen.size();
    RingMLANode.opDesc = {0, "RingAttentionOperation", RingMLAParam};
    ATB_LOG(INFO) << GetLogPrefix() << " update AsdOps::OpParam::RingMLAParam.type: " << RingMLAParam.type
                  << "headNum: " << param_.headNum << ", qSeqLen.size: " << RingMLAParam.qSeqLen.size()
                  << ", kvSeqLen.size: " << RingMLAParam.kvSeqLen.size() << ", qkScale: " << RingMLAParam.tor
                  << ", kvHead: " << RingMLAParam.kvHead << ", maskType: " << RingMLAParam.maskType;
                //   << ", headDimV: " << RingMLAParam.headDimV;
    return NO_ERROR;
}

void RingMLAOpsRunner::SetRingMLAParam(AtbOps::OpParam::RINGMLA &RingMLAParam)
{
    RingMLAParam.isRing = isInputSoftmaxLse_;
    RingMLAParam.headSize = param_.headNum;
    RingMLAParam.kvHead = param_.kvHeadNum;
    RingMLAParam.tor = param_.qkScale;
    RingMLAParam.type = AtbOps::OpParam::RINGMLA::Type::;
    if (param_.maskType == infer::RingMLAParam::MaskType::NO_MASK) {
        RingMLAParam.maskType = static_cast<AtbOps::OpParam::RINGMLA::MaskType>(
            AtbOps::OpParam::RINGMLA::MaskType::MASK_TYPE_NONE);
    } else if (param_.maskType == infer::RingMLAParam::MaskType::MASK_TYPE_TRIU) {
        RingMLAParam.maskType = static_cast<AtbOps::OpParam::RINGMLA::MaskType>(
            AtbOps::OpParam::RINGMLA::MaskType::MASK_TYPE_NORM);
        RingMLAParam.isTriuMask = 1;
    }
}

RingMLAOpsRunner::~RingMLAOpsRunner() {}
} // namespace atb
