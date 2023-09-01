/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "self_attention_kv_cache_fusion_ops_baichuan2_runner_910a.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

static const uint64_t IN_TENSOR_COUNT = 9;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 8;

namespace AclTransformer {
SelfAttentionKvCacheFusionOpsBaichuan2Runner910A::SelfAttentionKvCacheFusionOpsBaichuan2Runner910A(
    const SelfAttentionKvCacheFusionParam &param)
    : OpsRunner("SelfAttentionKvCacheFusionOpsBaichuan2Runner910A", RUNNER_TYPE_SELF_ATTENTION_KV_FUSION_CACHE),
      param_(param)
{
    setupCacheEnable_ = false;
    ASD_LOG(INFO) << "SelfAttentionKvCacheFusionOpsBaichuan2Runner910A new, setupCacheEnable:" << setupCacheEnable_;
    BuildGraphWithMuls();
    SetKernelGrapModifyFunc();
}

SelfAttentionKvCacheFusionOpsBaichuan2Runner910A::~SelfAttentionKvCacheFusionOpsBaichuan2Runner910A() {}

void SelfAttentionKvCacheFusionOpsBaichuan2Runner910A::BuildGraphWithMuls()
{
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);

    uint64_t tensorId = 0;
    // kv cache input
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(tensorId++);

    AsdOps::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
    // flash attention input
    AsdOps::Tensor &attentionMask = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &tokenOffset = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &layerId = kernelGraph_.inTensors.at(tensorId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    uint64_t outTensorId = 0;
    AsdOps::Tensor &contextOut = kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    uint64_t internalTensorId = 0;
    AsdOps::Tensor &transdataKey = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &transdataValue = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &transdataQuery = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &context = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(NODE_COUNT);
    size_t nodeId = 0;
    auto &kTransdataNode = kernelGraph_.nodes.at(nodeId++);
    auto &kCacheNode = kernelGraph_.nodes.at(nodeId++);
    auto &vTransdataNode = kernelGraph_.nodes.at(nodeId++);
    auto &vCacheNode = kernelGraph_.nodes.at(nodeId++);
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    auto &qTransdataNode = kernelGraph_.nodes.at(nodeId++);
    auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataContextNode = kernelGraph_.nodes.at(nodeId++);

    // transdata for k
    kTransdataNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    kTransdataNode.inTensors = {&mixedKey};
    kTransdataNode.outTensors = {&transdataKey};
    kTransdataNode.inTensorViewFuncs.resize(kTransdataNode.inTensors.size());
    kTransdataNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
    kTransdataNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        oriDimK_ = runInfo.GetInTensor(0).desc.dims;
        ASD_LOG(INFO) << "oriDimK_ shape is " << oriDimK_;
    };

    // k cache
    kCacheNode.opDesc = {0, "KVCacheOperation", AsdOps::OpParam::KVCache{AsdOps::OpParam::KVCache::KVCACHE}};
    kCacheNode.inTensors = {&transdataKey, &layerId, &cacheK, &tokenOffset, &seqLen};
    kCacheNode.outTensors = {&cacheK}; // Kcache and Vcache output and input use same space
    kCacheNode.inTensorViewFuncs.resize(kCacheNode.inTensors.size());
    kCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(1), oldDims.at(2) / 16, 16, oldDims.at(3)};
    };
    kCacheNode.inTensorViewFuncs[2] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(3) / 16, oldDims.at(2) / 16, 16, 16};
    };

    vTransdataNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    vTransdataNode.inTensors = {&mixedValue};
    vTransdataNode.outTensors = {&transdataValue};
    vTransdataNode.inTensorViewFuncs.resize(vTransdataNode.inTensors.size());
    vTransdataNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    // v cache
    vCacheNode.opDesc = {0, "KVCacheOperation", AsdOps::OpParam::KVCache{AsdOps::OpParam::KVCache::KVCACHE}};
    vCacheNode.inTensors = {&transdataValue, &layerId, &cacheV, &tokenOffset, &seqLen};
    vCacheNode.outTensors = {&cacheV}; // Kcache and Vcache output and input use same space
    vCacheNode.inTensorViewFuncs.resize(vCacheNode.inTensors.size());
    vCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(1), oldDims.at(2) / 16, 16, oldDims.at(3)};
    };
    vCacheNode.inTensorViewFuncs[2] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(3) / 16, oldDims.at(2) / 16, 16, 16};
    };

    //
    float varAttr = 1.0 / (sqrt(param_.dk) * (param_.layerId + 1));
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulsQNode.inTensors = {&mixedQuery};
    mulsQNode.outTensors = {&divOut};
    mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    qTransdataNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    qTransdataNode.inTensors = {&mixedQuery};
    qTransdataNode.outTensors = {&transdataQuery};
    qTransdataNode.inTensorViewFuncs.resize(qTransdataNode.inTensors.size());
    qTransdataNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    // 4、flash attention
    float tor = (float)(param_.layerId + 1);
    flashAttentionNode.opDesc = {0, "AttentionOperation",
                                 AsdOps::OpParam::Attention{param_.headNum, param_.seqLen, param_.tokenOffset, tor}};
    flashAttentionNode.inTensors = {&divOut, &cacheK, &cacheV, &layerId, &attentionMask};
    flashAttentionNode.outTensors = {&context};
    flashAttentionNode.inTensorViewFuncs.resize(flashAttentionNode.inTensors.size());
    flashAttentionNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                                 AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    transdataContextNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataContextNode.inTensors = {&context};
    transdataContextNode.outTensors = {&contextOut};
    transdataContextNode.inTensorViewFuncs.resize(transdataContextNode.inTensors.size());
    transdataContextNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transQKTargetDims = {oriDimK_.at(1), oriDimK_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transQKTargetDims})});
    };
}

void SelfAttentionKvCacheFusionOpsBaichuan2Runner910A::SetKernelGrapModifyFunc()
{
    kernelGraph_.kernelGraphModifyFunc = [&](const RunnerVariantPack &runnerVariantPack) {
        if (typeid(SelfAttentionKvCacheFusionVariantPackParam) != runnerVariantPack.param.Type()) {
            ASD_LOG(FATAL) << "SelfAttentionKvCacheFusionOpsBaichuan2Runner910A invalid type "
                              "SelfAttentionKvCacheFusionOpsBaichuan2Runner910A";
            return;
        }
        const SelfAttentionKvCacheFusionVariantPackParam &newParam =
            AsdOps::AnyCast<SelfAttentionKvCacheFusionVariantPackParam>(runnerVariantPack.param);
        const size_t flashAttentionNodeId = 6;
        auto &flashAttentionNode = kernelGraph_.nodes.at(flashAttentionNodeId);

        float tor = (float)(newParam.layerId + 1);
        flashAttentionNode.opDesc = {
            0, "AttentionOperation",
            AsdOps::OpParam::Attention{param_.headNum, newParam.seqLen, newParam.tokenOffset, tor}};
        ASD_LOG(INFO) << "SelfAttentionKvCacheFusionOpsBaichuan2Runner910A SetOpDesc AsdOps::OpParam::Attention.headNum:"
                      << param_.headNum << ", seqLen:" << newParam.seqLen << ", tokenOffset:" << newParam.tokenOffset
                      << ", tor:" << tor;
    };
}
} // namespace AclTransformer
