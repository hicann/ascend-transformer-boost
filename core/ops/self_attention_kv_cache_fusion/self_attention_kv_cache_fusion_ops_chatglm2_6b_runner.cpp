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
#include "self_attention_kv_cache_fusion_ops_chatglm2_6b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheFusionOpsChatGlm2Runner::SelfAttentionKvCacheFusionOpsChatGlm2Runner(
    const SelfAttentionKvCacheFusionParam &param)
    : OpsRunner("SelfAttentionKvCacheFusionOpsChatGlm2Runner", RUNNER_TYPE_SELF_ATTENTION_KV_FUSION_CACHE),
      param_(param)
{
    setupCacheEnable_ = false;
    ASD_LOG(INFO) << "SelfAttentionKvCacheFusionOpsChatGlm2Runner new, setupCacheEnable:" << setupCacheEnable_;
    BuildGraphWithMuls();
    SetKernelGrapModifyFunc();
}

SelfAttentionKvCacheFusionOpsChatGlm2Runner::~SelfAttentionKvCacheFusionOpsChatGlm2Runner() {}


void SelfAttentionKvCacheFusionOpsChatGlm2Runner::BuildGraphWithMuls()
{
    const size_t intTensorCount = 9;
    const size_t nodeCount = 4;
    kernelGraph_.inTensors.resize(intTensorCount);

    size_t tensorId = 0;
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

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);

    kernelGraph_.internalTensors.resize(1);
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(0);

    size_t nodeId = 0;
    kernelGraph_.nodes.resize(nodeCount);
    auto &kCacheNode = kernelGraph_.nodes.at(nodeId++);
    auto &vCacheNode = kernelGraph_.nodes.at(nodeId++);
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);

    int64_t np = param_.numHeadsPerPartition;
    int64_t gp = param_.numGroupsPerPartition;

    // 1、k cache
    kCacheNode.opDesc = {0, "KVCacheOperation", AsdOps::OpParam::KVCache{AsdOps::OpParam::KVCache::KVCACHE}};
    kCacheNode.inTensors = {&mixedKey, &layerId, &cacheK, &tokenOffset, &seqLen};
    kCacheNode.outTensors = {&cacheK}; // Kcache and Vcache output and input use same space
    kCacheNode.inTensorViewFuncs.resize(kCacheNode.inTensors.size());
    kCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    // 2、V cache  seq_len, batch, head_num, head_size]
    vCacheNode.opDesc = {0, "KVCacheOperation", AsdOps::OpParam::KVCache{AsdOps::OpParam::KVCache::KVCACHE}};
    vCacheNode.inTensors = {&mixedValue, &layerId, &cacheV, &tokenOffset, &seqLen};
    vCacheNode.outTensors = {&cacheV}; // Kcache and Vcache output and input use same space
    vCacheNode.inTensorViewFuncs.resize(vCacheNode.inTensors.size());
    vCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
    vCacheNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    // 3、div
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

    // 4、flash attention
    float tor = (float)(param_.layerId + 1);
    int kvHead = param_.numGroupsPerPartition;
    flashAttentionNode.opDesc = {0, "AttentionOperation",
                                 AsdOps::OpParam::Attention{param_.headNum, param_.seqLen, param_.tokenOffset, tor, kvHead}};
    flashAttentionNode.inTensors = {&divOut, &cacheK, &cacheV, &layerId, &attentionMask};
    flashAttentionNode.outTensors = {&context};
    flashAttentionNode.inTensorViewFuncs.resize(flashAttentionNode.inTensors.size());
    flashAttentionNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                                 AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
}

void SelfAttentionKvCacheFusionOpsChatGlm2Runner::SetKernelGrapModifyFunc()
{
    kernelGraph_.kernelGraphModifyFunc = [&](const RunnerVariantPack &runnerVariantPack) {
        if (typeid(SelfAttentionKvCacheFusionVariantPackParam) != runnerVariantPack.param.Type()) {
            ASD_LOG(FATAL) << "SelfAttentionKvCacheFusionOpsChatGlm2Runner invalid type "
                              "SelfAttentionKvCacheFusionVariantPackParam";
            return;
        }
        const SelfAttentionKvCacheFusionVariantPackParam &newParam =
            AsdOps::AnyCast<SelfAttentionKvCacheFusionVariantPackParam>(runnerVariantPack.param);
        const size_t flashAttentionNodeId = 3;
        auto &flashAttentionNode = kernelGraph_.nodes.at(flashAttentionNodeId);

        float tor = (float)(newParam.layerId + 1);
        int kvHead = param_.numGroupsPerPartition;
        flashAttentionNode.opDesc = {
            0, "AttentionOperation",
            AsdOps::OpParam::Attention{param_.headNum, newParam.seqLen, newParam.tokenOffset, tor, kvHead}};
        ASD_LOG(INFO) << "SelfAttentionKvCacheFusionOpsChatGlm2Runner SetOpDesc AsdOps::OpParam::Attention.headNum:"
                      << param_.headNum << ", seqLen:" << newParam.seqLen << ", tokenOffset:" << newParam.tokenOffset;
    };
}
} // namespace AclTransformer