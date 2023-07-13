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
const size_t IN_TENSOR_COUNT = 9;
const size_t NODE_COUNT = 4;

SelfAttentionKvCacheFusionOpsChatGlm6bRunner::SelfAttentionKvCacheFusionOpsChatGlm6bRunner(
    const SelfAttentionKvCacheFusionParam &param)
    : OpsRunner("SelfAttentionKvCacheFusionOpsChatGlm6bRunner", RUNNER_TYPE_SELF_ATTENTION_KV_FUSION_CACHE),
      param_(param)
{
    ASD_LOG(INFO)
        << "SelfAttentionKvCacheFusionOpsChatGlm6bRunner::SelfAttentionKvCacheFusionOpsChatGlm6bRunner called";

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);

    size_t tensorId = 0;
    // kv cache input
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
    // flash attention input
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &attentionMask = kernelGraph_.inTensors.at(tensorId++);

    AsdOps::Tensor &tokenOffset = kernelGraph_.inTensors.at(tensorId++);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(tensorId++);

    AsdOps::Tensor &layerId = kernelGraph_.inTensors.at(tensorId++);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);

    kernelGraph_.internalTensors.resize(1);
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(0);

    size_t nodeId = 0;
    kernelGraph_.nodes.resize(NODE_COUNT);
    auto &KCacheNode = kernelGraph_.nodes.at(nodeId++);
    auto &VCacheNode = kernelGraph_.nodes.at(nodeId++);
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);

    // 1、k cache
    KCacheNode.opDesc = {0, "KVCacheOperation"};
    KCacheNode.inTensors = {&mixedKey, &layerId, &cacheK, &tokenOffset, &seqLen};
    KCacheNode.outTensors = {&cacheK}; // Kcache and Vcache output and input use same space
    KCacheNode.inTensorViewFuncs.resize(KCacheNode.inTensors.size());
    KCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
    KCacheNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    // 2、V cache  seq_len, batch, head_num, head_size]
    VCacheNode.opDesc = {0, "KVCacheOperation"};
    VCacheNode.inTensors = {&mixedValue, &layerId, &cacheV, &tokenOffset, &seqLen};
    VCacheNode.outTensors = {&cacheV}; // Kcache and Vcache output and input use same space
    VCacheNode.inTensorViewFuncs.resize(VCacheNode.inTensors.size());
    VCacheNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
    VCacheNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    // 3、div
    float varAttr = 1.0 / (sqrt(param.dk) * (param.layerId + 1));
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulsQNode.inTensors = {&mixedQuery};
    mulsQNode.outTensors = {&divOut};
    mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    ASD_LOG(INFO) << GetName() << " AsdOps::OpParam::Attention param headNum:" << param.headNum
                  << ", seqLen:" << param.seqLen << ", tokenOffset:" << param.tokenOffset;
    // 4、flash attention
    flashAttentionNode.opDesc = {0, "AttentionOperation",
                                 AsdOps::OpParam::Attention{param.headNum, param.seqLen, param.tokenOffset}};
    flashAttentionNode.inTensors = {&divOut, &cacheK, &cacheV, &layerId, &attentionMask};
    flashAttentionNode.outTensors = {&context};
    flashAttentionNode.inTensorViewFuncs.resize(flashAttentionNode.inTensors.size());
    flashAttentionNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                                 AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
}

SelfAttentionKvCacheFusionOpsChatGlm6bRunner::~SelfAttentionKvCacheFusionOpsChatGlm6bRunner() {}
} // namespace AclTransformer
