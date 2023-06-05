/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "self_attention_kv_cache_torch_runner.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/utils/tensor_cache.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <ATen/ATen.h>
#include <cmath>

namespace AclTransformer {
SelfAttentionKvCacheTorchRunner::SelfAttentionKvCacheTorchRunner(const SelfAttentionKvCacheParam &param)
    : Runner("SelfAttentionKvCacheTorchRunner"), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKvCacheOperation::SelfAttentionKvCacheOperation called";
}

SelfAttentionKvCacheTorchRunner::~SelfAttentionKvCacheTorchRunner() {}

AsdOps::Status SelfAttentionKvCacheTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    // in : Q K V attention_mast pastK pastV
    // out : result presentK presentV
    // Q K V pastK pastV : [seq_len, batch, head_num, head_size]
    ASD_LOG(INFO) << "headNum:" << this->param_.headNum << "   dk: " << this->param_.dk
                  << "  layer id: " << this->param_.layerId;
    torch::Tensor mixedQuery = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[0].data);
    torch::Tensor mixedKey = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[1].data);
    torch::Tensor mixedValue = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[2].data);
    torch::Tensor attention_mask = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[3].data);
    torch::Tensor pastKey = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[4].data);
    torch::Tensor pastValue = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[5].data);
    ASD_LOG(INFO) << "start";
    ASD_LOG(INFO) << "mixedQuery" << mixedQuery.sizes();
    ASD_LOG(INFO) << "mixedKey" << mixedKey.sizes();
    ASD_LOG(INFO) << "mixedValue" << mixedValue.sizes();
    ASD_LOG(INFO) << "attention_mask" << attention_mask.sizes();
    ASD_LOG(INFO) << "pastKey" << pastKey.sizes();
    ASD_LOG(INFO) << "pastValue" << pastValue.sizes();

    torch::Tensor presentKey = torch::cat({pastKey, mixedKey}, 0);
    torch::Tensor *presentKeyout = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.outTensors[1].data);
    *presentKeyout = presentKey;
    ASD_LOG(INFO) << "cat K end";
    mixedQuery =
        mixedQuery.view({mixedQuery.sizes()[0], mixedQuery.sizes()[1] * mixedQuery.sizes()[2], mixedQuery.sizes()[3]});
    mixedQuery = torch::transpose(mixedQuery, 0, 1);

    torch::Tensor presentValue = torch::cat({pastValue, mixedValue}, 0);
    torch::Tensor *presentValueOut = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.outTensors[2].data);
    *presentValueOut = presentValue;
    ASD_LOG(INFO) << "cat V end";
    presentValue = presentValue.view(
        {presentValue.sizes()[0], presentValue.sizes()[1] * presentValue.sizes()[2], presentValue.sizes()[3]});
    presentValue = torch::transpose(presentValue, 0, 1);

    presentKey =
        presentKey.view({presentKey.sizes()[0], presentKey.sizes()[1] * presentKey.sizes()[2], presentKey.sizes()[3]});
    presentKey = presentKey.permute({1, 2, 0});

    double scal = 1 / (sqrt(this->param_.dk) * (this->param_.layerId + 1));
    mixedQuery = torch::mul(mixedQuery, scal);
    // [b, head_num, sq, sk]
    torch::Tensor attentionScores = torch::bmm(mixedQuery, presentKey).contiguous();
    if (attention_mask.sum().item<bool>() > 0) {
        attentionScores.masked_fill_(attention_mask, -10000.0);
    }
    ASD_LOG(INFO) << "bmm1 end";
    // to float?
    attentionScores = torch::mul(attentionScores, this->param_.layerId + 1.0);
    torch::Tensor attention_probs = torch::softmax(attentionScores, -1);
    ASD_LOG(INFO) << "softmax end";
    torch::Tensor contextLayer = torch::bmm(attention_probs, presentValue);
    ASD_LOG(INFO) << "bmm2 end";
    contextLayer = torch::transpose(contextLayer, 0, 1).contiguous();

    torch::Tensor *atOutTensor = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.outTensors[0].data);
    *atOutTensor = contextLayer;

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer