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
#include "self_attention_kv_cache_fusion_ops_chatglm6b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheFusionOpsChatGlm6bRunner::
    SelfAttentionKvCacheFusionOpsChatGlm6bRunner(const SelfAttentionKvCacheFusionParam &param)
    : OpsRunner("SelfAttentionKvCacheFusionOpsChatGlm6bRunner", RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param)
{
    ASD_LOG(INFO) <<
        "SelfAttentionKvCacheFusionOpsChatGlm6bRunner::SelfAttentionKvCacheFusionOpsChatGlm6bRunner called";

    kernelGraph_.inTensors.resize(inTensSize);
    // kv cache input
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &cacheK = kernelGraph_.inTensors.at(index2);
    AsdOps::Tensor &cacheV = kernelGraph_.inTensors.at(index3);
    // flash attention input
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(index4);
    AsdOps::Tensor &attentionMask = kernelGraph_.inTensors.at(index5);

    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(index6);
    AsdOps::Tensor &tokenOffset = kernelGraph_.inTensors.at(index7);
    AsdOps::Tensor &layerId = kernelGraph_.inTensors.at(index8);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);
    
    kernelGraph_.nodes.resize(nodeSize);
    auto &KCacheNode = kernelGraph_.nodes.at(0);
    auto &VCacheNode = kernelGraph_.nodes.at(1);
    auto &flashAttentionNode = kernelGraph_.nodes.at(index2);

    // 1、k cache
    KCacheNode.opDesc = {0, "KvCacheOperation"};
    KCacheNode.inTensors = {&mixedKey, &layerId, &cacheK, &seqLen, &tokenOffset};
    KCacheNode.outTensors = {&cacheK}; // Kcache and Vcache output and input use same space

    // 2、V cache
    VCacheNode.opDesc = {0, "KvCacheOperation"};
    VCacheNode.inTensors = {&mixedValue, &layerId, &cacheV, &seqLen, &tokenOffset};
    KCacheNode.outTensors = {&cacheV}; // Kcache and Vcache output and input use same space

    // 3、flash attention
    flashAttentionNode.opDesc = {0, "FlashAttentionOperation"};
    flashAttentionNode.inTensors = {&mixedQuery, &cacheK, &cacheV, &seqLen, &tokenOffset, &layerId, &attentionMask};
    flashAttentionNode.outTensors = {&context};
}

SelfAttentionKvCacheFusionOpsChatGlm6bRunner::~SelfAttentionKvCacheFusionOpsChatGlm6bRunner() {}
} // namespace AclTransformer
