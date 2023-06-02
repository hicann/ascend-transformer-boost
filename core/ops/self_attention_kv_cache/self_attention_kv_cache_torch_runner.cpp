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

AsdOps::Status SelfAttentionKvCacheTorchRunner::Execute(Handle &handle, VariantPack &variantPack)
{ 
    // in : Q K V attention_mast pastK pastV
    // out : result presentK presentV
    // Q K pastK pastV : [seq_len, batch, head_num, head_size]
    // V : [seq_len, batch, all_head_size]
    ASD_LOG(INFO) << "headNum:" << this->param_.headNum << "   dk: " << this->param_.dk << "  layer id: " << this->param_.layerId;
    torch::Tensor mixedQuery = AsdOpsTensor2AtTensor(variantPack.inTensors[0]);
    torch::Tensor mixedKey = AsdOpsTensor2AtTensor(variantPack.inTensors[1]);
    torch::Tensor mixedValue = AsdOpsTensor2AtTensor(variantPack.inTensors[2]);
    torch::Tensor attention_mask = AsdOpsTensor2AtTensor(variantPack.inTensors[3]);
    torch::Tensor pastKey = AsdOpsTensor2AtTensor(variantPack.inTensors[4]);
    torch::Tensor pastValue = AsdOpsTensor2AtTensor(variantPack.inTensors[5]);
    ASD_LOG(INFO) << "start";
    ASD_LOG(INFO) << "mixedQuery" << mixedQuery.sizes();
    ASD_LOG(INFO) << "mixedKey" << mixedKey.sizes();
    ASD_LOG(INFO) << "mixedValue" << mixedValue.sizes();
    ASD_LOG(INFO) << "attention_mask" << attention_mask.sizes();
    ASD_LOG(INFO) << "pastKey" << pastKey.sizes();
    ASD_LOG(INFO) << "pastValue" << pastValue.sizes();

    mixedKey = torch::cat({pastKey, mixedKey}, 0);
    ASD_LOG(INFO) << "cat K end";
    mixedQuery = mixedQuery.view({mixedQuery.sizes()[0], mixedQuery.sizes()[1] * mixedQuery.sizes()[2], mixedQuery.sizes()[3]});
    mixedQuery = torch::transpose(mixedQuery, 0, 1);

    mixedValue = mixedValue.view({mixedValue.sizes()[0], mixedValue.sizes()[1], this->param_.headNum,
                                  mixedValue.sizes()[2] / this->param_.headNum});
    mixedValue = torch::cat({pastValue, mixedValue}, 0); 
    ASD_LOG(INFO) << "cat V end";         
    mixedValue = mixedValue.view({mixedValue.sizes()[0], mixedValue.sizes()[1] * mixedValue.sizes()[2], mixedValue.sizes()[3]});               
    mixedValue = torch::transpose(mixedValue, 0, 1);

    mixedKey = mixedKey.view({mixedKey.sizes()[0], mixedKey.sizes()[1] * mixedKey.sizes()[2], mixedKey.sizes()[3]});
    mixedKey = mixedKey.permute({1, 2, 0});

    double scal = 1 / (sqrt(this->param_.dk) * (this->param_.layerId + 1));
    mixedQuery = torch::mul(mixedQuery, scal);
    // [b, head_num, sq, sk]
    torch::Tensor attentionScores = torch::bmm(mixedQuery, mixedKey).contiguous();
    if (attention_mask.sum().item<bool>() > 0) {
        attentionScores.masked_fill_(attention_mask, -10000.0);
    }
    ASD_LOG(INFO) << "bmm1 end";
    // to float?
    attentionScores = torch::mul(attentionScores, this->param_.layerId + 1.0);
    torch::Tensor attention_probs = torch::softmax(attentionScores, -1);
    ASD_LOG(INFO) << "softmax end";
    torch::Tensor contextLayer = torch::bmm(attention_probs, mixedValue);
    ASD_LOG(INFO) << "bmm2 end";
    contextLayer = torch::transpose(contextLayer, 0, 1).contiguous();
    contextLayer = contextLayer
                       .view({contextLayer.sizes()[0], contextLayer.sizes()[1] / this->param_.headNum,
                              contextLayer.sizes()[2] * this->param_.headNum})
                       .contiguous();

    int ret = AsdRtMemCopyAsync(variantPack.outTensors[0].data, variantPack.outTensors[0].dataSize,
                                contextLayer.storage().data_ptr().get(), variantPack.outTensors[0].dataSize,
                                ASDRT_MEMCOPY_DEVICE_TO_DEVICE, handle.stream);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail";

    ret = AsdRtMemCopyAsync(variantPack.outTensors[1].data, variantPack.outTensors[1].dataSize,
                                mixedKey.storage().data_ptr().get(), variantPack.outTensors[1].dataSize,
                                ASDRT_MEMCOPY_DEVICE_TO_DEVICE, handle.stream);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail";

    ret = AsdRtMemCopyAsync(variantPack.outTensors[2].data, variantPack.outTensors[2].dataSize,
                                mixedValue.storage().data_ptr().get(), variantPack.outTensors[2].dataSize,
                                ASDRT_MEMCOPY_DEVICE_TO_DEVICE, handle.stream);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail";
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer