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
    if (0) {
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
        torch::Tensor *atOutTensor = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.outTensors[0].data);
        torch::Tensor *presentKeyout = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.outTensors[1].data);
        torch::Tensor *presentValueOut = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.outTensors[2].data);
        ASD_LOG(INFO) << "start";
        ASD_LOG(INFO) << "mixedQuery" << mixedQuery.sizes();
        ASD_LOG(INFO) << "mixedKey" << mixedKey.sizes();
        ASD_LOG(INFO) << "mixedValue" << mixedValue.sizes();
        ASD_LOG(INFO) << "attention_mask" << attention_mask.sizes();
        ASD_LOG(INFO) << "pastKey" << pastKey.sizes();
        ASD_LOG(INFO) << "pastValue" << pastValue.sizes();

        torch::Tensor presentKey = torch::cat({pastKey, mixedKey}, 0);
        // torch::save(presentKey.to(at::Device(at::kCPU)), "cat_presentKey.pth");
        *presentKeyout = presentKey;
        ASD_LOG(INFO) << "cat K end" << presentKey.sizes();
        // [seq_len, batch*head_num, head_size]
        mixedQuery = mixedQuery.view(
            {mixedQuery.sizes()[0], mixedQuery.sizes()[1] * mixedQuery.sizes()[2], mixedQuery.sizes()[3]});
        // [batch*head_num, seq_len, head_size]
        mixedQuery = torch::transpose(mixedQuery, 0, 1);

        torch::Tensor presentValue = torch::cat({pastValue, mixedValue}, 0);
        *presentValueOut = presentValue;
        ASD_LOG(INFO) << "cat V end" << presentValue.sizes();
        // [seq_len, batch*head_num, head_size]
        presentValue = presentValue.view(
            {presentValue.sizes()[0], presentValue.sizes()[1] * presentValue.sizes()[2], presentValue.sizes()[3]});
        // [batch*head_num, seq_len, head_size]
        presentValue = torch::transpose(presentValue, 0, 1);
        // [seq_len, batch*head_num, head_size]
        presentKey = presentKey.view(
            {presentKey.sizes()[0], presentKey.sizes()[1] * presentKey.sizes()[2], presentKey.sizes()[3]});
        // [batch*head_num, head_size, seq_len]
        presentKey = presentKey.permute({1, 2, 0});

        mixedQuery = mixedQuery / (sqrt(presentValue.sizes()[2]) * (float)(this->param_.layerId + 1));
        // [batch*head_num, seq_len_q, seq_len_k]
        // torch::save(mixedQuery.to(at::Device(at::kCPU)), "mixedQuery.pth");
        // torch::save(presentKey.to(at::Device(at::kCPU)), "presentKey.pth");
        torch::Tensor attentionScores = torch::bmm(mixedQuery, presentKey).contiguous();

        // [b, head_num, sq, sk]
        attentionScores = attentionScores.view({attentionScores.sizes()[0] / this->param_.headNum, this->param_.headNum,
                                                attentionScores.sizes()[1], attentionScores.sizes()[2]});
        // torch::save(attentionScores.to(at::Device(at::kCPU)), "bmm1.pth");
        ASD_LOG(INFO) << "attentionScores:" << attentionScores.sizes();
        if (attention_mask.sum().item<int64_t>() > 0) {
            // torch::save(attentionScores.to(at::Device(at::kCPU)), "before_mask_fill.pth");
            // torch::save(attention_mask.to(at::Device(at::kCPU)), "attention_mask.pth");
            attentionScores.masked_fill_(attention_mask, -10000.0);
            // torch::save(attentionScores.to(at::Device(at::kCPU)), "mask_fill.pth");
        }
        ASD_LOG(INFO) << "bmm1 end";
        attentionScores = attentionScores.to(torch::kFloat32);
        // torch::save(attentionScores.to(at::Device(at::kCPU)), "before_mul2.pth");
        attentionScores = torch::mul(attentionScores, this->param_.layerId + 1.0);
        // torch::save(attentionScores.to(at::Device(at::kCPU)), "mul2.pth");
        torch::Tensor attention_probs = torch::softmax(attentionScores, -1);
        attention_probs = attention_probs.to(torch::kHalf);
        // torch::save(attention_probs.to(at::Device(at::kCPU)), "softmax.pth");
        ASD_LOG(INFO) << "softmax end";
        // [batch*head_num, seq_len_q, seq_len_k]
        attention_probs = attention_probs.view({attentionScores.sizes()[0] * attentionScores.sizes()[1],
                                                attentionScores.sizes()[2], attentionScores.sizes()[3]});
        // [batch*head_num, seq_len_q, head_size]
        // torch::save(attention_probs.to(at::Device(at::kCPU)), "m_before_bmm2.pth");
        // torch::save(presentValue.to(at::Device(at::kCPU)), "n_before_bmm2.pth");
        torch::Tensor contextLayer = torch::bmm(attention_probs, presentValue);
        // torch::save(contextLayer.to(at::Device(at::kCPU)), "bmm2.pth");
        ASD_LOG(INFO) << "bmm2 end";
        ASD_LOG(INFO) << "contextLayer" << contextLayer.sizes();
        // [seq_len_q, batch*head_num, head_size]
        contextLayer = torch::transpose(contextLayer, 0, 1);
        ASD_LOG(INFO) << "contextLayer" << contextLayer.sizes();
        contextLayer = contextLayer
                           .view({contextLayer.sizes()[0], contextLayer.sizes()[1] / this->param_.headNum,
                                  this->param_.headNum, contextLayer.sizes()[2]})
                           .contiguous();
        ASD_LOG(INFO) << "contextLayer" << contextLayer.sizes();
        contextLayer = contextLayer.view(
            {contextLayer.sizes()[0], contextLayer.sizes()[1], contextLayer.sizes()[2] * contextLayer.sizes()[3]});
        ASD_LOG(INFO) << "contextLayer" << contextLayer.sizes();
        if (contextLayer.sizes() != (*atOutTensor).sizes()) {
            ASD_LOG(ERROR) << "infer shape error" << (*atOutTensor).sizes();
        }
        *atOutTensor = contextLayer;
        // torch::save((*atOutTensor).to(at::Device(at::kCPU)), "final.pth");
    } else {
        // in : Q K V attention_mast pastK pastV
        // out : result presentK presentV
        // Q K V pastK pastV : [seq_len, batch, head_num, head_size]
        ASD_LOG(INFO) << "headNum:" << this->param_.headNum << "   dk: " << this->param_.dk
                      << "  layer id: " << this->param_.layerId;
        torch::Tensor mixedQuery = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[0]);
        torch::Tensor mixedKey = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[1]);
        torch::Tensor mixedValue = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[2]);
        torch::Tensor attention_mask = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[3]);
        torch::Tensor pastKey = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[4]);
        torch::Tensor pastValue = AsdOpsTensor2AtTensor(handle, variantPack.inTensors[5]);
        // torch::Tensor *atOutTensor = AsdOpsTensor2AtTensor(handle, variantPack.outTensors[0]);
        // torch::Tensor *presentKeyout = AsdOpsTensor2AtTensor(handle, variantPack.outTensors[1]);
        // torch::Tensor *presentValueOut = AsdOpsTensor2AtTensor(handle, variantPack.outTensors[2]);
        ASD_LOG(INFO) << "start";
        ASD_LOG(INFO) << "mixedQuery" << mixedQuery.sizes();
        ASD_LOG(INFO) << "mixedKey" << mixedKey.sizes();
        ASD_LOG(INFO) << "mixedValue" << mixedValue.sizes();
        ASD_LOG(INFO) << "attention_mask" << attention_mask.sizes();
        ASD_LOG(INFO) << "pastKey" << pastKey.sizes();
        ASD_LOG(INFO) << "pastValue" << pastValue.sizes();

        torch::Tensor presentKey = torch::cat({pastKey, mixedKey}, 0).contiguous();
        // torch::save(presentKey.to(at::Device(at::kCPU)), "cat_presentKey.pth");
        //*presentKeyout = presentKey;
        ASD_LOG(INFO) << "presentKey.sizes:" << presentKey.sizes() << ", variantPack.outTensors[1].desc:"
                      << AsdOpsTensorDescToString(variantPack.outTensors[1].desc);
        CopyAtTensor2AsdOpsTensor(handle.stream, presentKey, variantPack.outTensors[1]);
        ASD_LOG(INFO) << "cat K end" << presentKey.sizes();
        // [seq_len, batch*head_num, head_size]
        mixedQuery = mixedQuery.view(
            {mixedQuery.sizes()[0], mixedQuery.sizes()[1] * mixedQuery.sizes()[2], mixedQuery.sizes()[3]});
        // [batch*head_num, seq_len, head_size]
        mixedQuery = torch::transpose(mixedQuery, 0, 1);

        torch::Tensor presentValue = torch::cat({pastValue, mixedValue}, 0).contiguous();
        //*presentValueOut = presentValue;
        ASD_LOG(INFO) << "presentValue.sizes:" << presentValue.sizes() << ", variantPack.outTensors[2].desc:"
                      << AsdOpsTensorDescToString(variantPack.outTensors[2].desc);
        CopyAtTensor2AsdOpsTensor(handle.stream, presentValue, variantPack.outTensors[2]);
        ASD_LOG(INFO) << "cat V end" << presentValue.sizes();
        // [seq_len, batch*head_num, head_size]
        presentValue = presentValue.view(
            {presentValue.sizes()[0], presentValue.sizes()[1] * presentValue.sizes()[2], presentValue.sizes()[3]});
        // [batch*head_num, seq_len, head_size]
        presentValue = torch::transpose(presentValue, 0, 1);
        // [seq_len, batch*head_num, head_size]
        presentKey = presentKey.view(
            {presentKey.sizes()[0], presentKey.sizes()[1] * presentKey.sizes()[2], presentKey.sizes()[3]});
        // [batch*head_num, head_size, seq_len]
        presentKey = presentKey.permute({1, 2, 0});

        mixedQuery = mixedQuery / (sqrt(presentValue.sizes()[2]) * (float)(this->param_.layerId + 1));
        // [batch*head_num, seq_len_q, seq_len_k]
        // torch::save(mixedQuery.to(at::Device(at::kCPU)), "mixedQuery.pth");
        // torch::save(presentKey.to(at::Device(at::kCPU)), "presentKey.pth");
        torch::Tensor attentionScores = torch::bmm(mixedQuery, presentKey).contiguous();

        // [b, head_num, sq, sk]
        attentionScores = attentionScores.view({attentionScores.sizes()[0] / this->param_.headNum, this->param_.headNum,
                                                attentionScores.sizes()[1], attentionScores.sizes()[2]});
        // torch::save(attentionScores.to(at::Device(at::kCPU)), "bmm1.pth");
        ASD_LOG(INFO) << "attentionScores:" << attentionScores.sizes();
        if (attention_mask.sum().item<int64_t>() > 0) {
            // torch::save(attentionScores.to(at::Device(at::kCPU)), "before_mask_fill.pth");
            // torch::save(attention_mask.to(at::Device(at::kCPU)), "attention_mask.pth");
            attentionScores.masked_fill_(attention_mask, -10000.0);
            // torch::save(attentionScores.to(at::Device(at::kCPU)), "mask_fill.pth");
        }
        ASD_LOG(INFO) << "bmm1 end";
        attentionScores = attentionScores.to(torch::kFloat32);
        // torch::save(attentionScores.to(at::Device(at::kCPU)), "before_mul2.pth");
        attentionScores = torch::mul(attentionScores, this->param_.layerId + 1.0);
        // torch::save(attentionScores.to(at::Device(at::kCPU)), "mul2.pth");
        torch::Tensor attention_probs = torch::softmax(attentionScores, -1);
        attention_probs = attention_probs.to(torch::kHalf);
        // torch::save(attention_probs.to(at::Device(at::kCPU)), "softmax.pth");
        ASD_LOG(INFO) << "softmax end";
        // [batch*head_num, seq_len_q, seq_len_k]
        attention_probs = attention_probs.view({attentionScores.sizes()[0] * attentionScores.sizes()[1],
                                                attentionScores.sizes()[2], attentionScores.sizes()[3]});
        // [batch*head_num, seq_len_q, head_size]
        // torch::save(attention_probs.to(at::Device(at::kCPU)), "m_before_bmm2.pth");
        // torch::save(presentValue.to(at::Device(at::kCPU)), "n_before_bmm2.pth");
        torch::Tensor contextLayer = torch::bmm(attention_probs, presentValue);
        // torch::save(contextLayer.to(at::Device(at::kCPU)), "bmm2.pth");
        ASD_LOG(INFO) << "bmm2 end";
        ASD_LOG(INFO) << "contextLayer" << contextLayer.sizes();
        // [seq_len_q, batch*head_num, head_size]
        contextLayer = torch::transpose(contextLayer, 0, 1);
        ASD_LOG(INFO) << "contextLayer" << contextLayer.sizes();
        contextLayer = contextLayer
                           .view({contextLayer.sizes()[0], contextLayer.sizes()[1] / this->param_.headNum,
                                  this->param_.headNum, contextLayer.sizes()[2]})
                           .contiguous();
        ASD_LOG(INFO) << "contextLayer" << contextLayer.sizes();
        contextLayer = contextLayer
                           .view({contextLayer.sizes()[0], contextLayer.sizes()[1],
                                  contextLayer.sizes()[2] * contextLayer.sizes()[3]})
                           .contiguous();
        ASD_LOG(INFO) << "contextLayer" << contextLayer.sizes() << ", variantPack.outTensors[0].desc:"
                      << AsdOpsTensorDescToString(variantPack.outTensors[0].desc);
        // if (contextLayer.sizes() != (*atOutTensor).sizes()) {
        //     ASD_LOG(ERROR) << "infer shape error" << (*atOutTensor).sizes();
        // }
        //*atOutTensor = contextLayer;
        // torch::save((*atOutTensor).to(at::Device(at::kCPU)), "final.pth");
        CopyAtTensor2AsdOpsTensor(handle.stream, contextLayer, variantPack.outTensors[0]);
    }

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer