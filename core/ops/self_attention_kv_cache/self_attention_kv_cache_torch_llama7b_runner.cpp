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
#include "self_attention_kv_cache_torch_llama7b_runner.h"
#include <cmath>
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheTorchLlama7bRunner::SelfAttentionKvCacheTorchLlama7bRunner(const SelfAttentionKvCacheParam &param)
    : Runner("SelfAttentionKvCacheTorchLlama7bRunner"), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKvCacheOperation::SelfAttentionKvCacheOperation called";
}

SelfAttentionKvCacheTorchLlama7bRunner::~SelfAttentionKvCacheTorchLlama7bRunner() {}

AsdOps::Status SelfAttentionKvCacheTorchLlama7bRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
#ifdef USE_TORCH_RUNNER
    // in : Q K V attention_mast pastK pastV
    // out : result presentK presentV
    // Q K V pastK pastV : [seq_len, batch, head_num, head_size]
    ASD_LOG(INFO) << "headNum:" << this->param_.headNum << "   dk: " << this->param_.dk;
    torch::Tensor mixedQuery = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[0]);
    torch::Tensor mixedKey = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[1]);
    torch::Tensor mixedValue = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[2]);
    torch::Tensor attention_mask = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[3]);
    torch::Tensor pastKey = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[4]);
    torch::Tensor pastValue = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[5]);
    // torch::Tensor *atOutTensor = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.outTensors[0]);
    // torch::Tensor *presentKeyout = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.outTensors[1]);
    // torch::Tensor *presentValueOut = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.outTensors[2]);
    ASD_LOG(INFO) << "start";
    ASD_LOG(INFO) << "mixedQuery" << mixedQuery.sizes();
    ASD_LOG(INFO) << "mixedKey" << mixedKey.sizes();
    ASD_LOG(INFO) << "mixedValue" << mixedValue.sizes();
    ASD_LOG(INFO) << "attention_mask" << attention_mask.sizes();
    ASD_LOG(INFO) << "pastKey" << pastKey.sizes();
    ASD_LOG(INFO) << "pastValue" << pastValue.sizes();

    // [batch, head_num, seq_len, head_size]
    mixedQuery = mixedQuery.permute({1, 2, 0, 3});
    mixedKey = mixedKey.permute({1, 2, 0, 3});
    mixedValue = mixedValue.permute({1, 2, 0, 3});
    pastKey = pastKey.permute({1, 2, 0, 3});
    pastValue = pastValue.permute({1, 2, 0, 3});

    ASD_LOG(INFO) << "cat pK pV";
    // [batch, head_num, kv_seq_len, head_size]
    torch::Tensor presentKey = torch::cat({pastKey, mixedKey}, 2).contiguous();
    ASD_LOG(INFO) << "cat pK ok";
    torch::Tensor presentValue = torch::cat({pastValue, mixedValue}, 2).contiguous();
    ASD_LOG(INFO) << "cat pV ok";
    // torch::save(presentKey.to(at::Device(at::kCPU)), "cat_presentKey.pth");
    //*presentKeyout = presentKey;

    torch::Tensor presentKeyOut = presentKey;
    // [seq_len, batch, head_num, head_size]
    presentKeyOut = presentKeyOut.permute({2, 0, 1, 3}).contiguous();
    if (!TorchUtil::IsTensorDimEqual(presentKeyOut.sizes(), variantPack.outTensors[1].desc.dims)) {
        ASD_LOG(ERROR) << "presentKeyOut.sizes:" << presentKeyOut.sizes() << ", variantPack.outTensors[2].desc:"
                       << TensorUtil::AsdOpsTensorDescToString(variantPack.outTensors[1].desc);
    }
    torch::save(presentKeyOut.to(at::Device(at::kCPU)), "presentKeyOut_example.path");
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, presentKeyOut, variantPack.outTensors[1]);

    torch::Tensor presentValueOut = presentValue;
    // [seq_len, batch, head_num, head_size]
    presentValueOut = presentValueOut.permute({2, 0, 1, 3}).contiguous();
    if (!TorchUtil::IsTensorDimEqual(presentValueOut.sizes(), variantPack.outTensors[2].desc.dims)) {
        ASD_LOG(ERROR) << "presentValueOut.sizes:" << presentValueOut.sizes() << ", variantPack.outTensors[2].desc:"
                       << TensorUtil::AsdOpsTensorDescToString(variantPack.outTensors[2].desc);
    }
    torch::save(presentValueOut.to(at::Device(at::kCPU)), "presentValueOut_example.path");
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, presentValueOut, variantPack.outTensors[2]);

    // [batch, head_num, head_size, kv_seq_len]
    presentKey = torch::transpose(presentKey, 2, 3);
    // [batch*head_num, seq_len, k_seq_len]
    mixedQuery =
        mixedQuery.view({mixedQuery.sizes()[0] * mixedQuery.sizes()[1], mixedQuery.sizes()[2], mixedQuery.sizes()[3]});
    presentKey =
        presentKey.view({presentKey.sizes()[0] * presentKey.sizes()[1], presentKey.sizes()[2], presentKey.sizes()[3]});
    ASD_LOG(INFO) << "mixedQuery before bmm1" << mixedQuery.sizes() << " dtype:" << mixedQuery.dtype();
    ASD_LOG(INFO) << "presentKey before bmm1" << presentKey.sizes() << " dtype:" << presentKey.dtype();
    // [batch*head_num, seq_len, k_seq_len]
    torch::Tensor attentionScores = torch::bmm(mixedQuery, presentKey);
    ASD_LOG(INFO) << "bmm1 ok";
    attentionScores = attentionScores / sqrt(this->param_.dk);
    torch::save(attentionScores.to(at::Device(at::kCPU)), "bmm1_div_out_example.path");
    ASD_LOG(INFO) << "div ok";

    // [batch*head_num, seq_len, k_seq_len]
    attention_mask = attention_mask.view(
        {attention_mask.sizes()[0] * attention_mask.sizes()[1], attention_mask.sizes()[2], attention_mask.sizes()[3]});
    attentionScores = attentionScores + attention_mask;
    ASD_LOG(INFO) << "add attention mask ok";

    // TODO: attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min,
    // device=attn_weights.device)

    // [batch, head_num, seq_len, k_seq_len]
    // caffe2::TypeMeta attentionScoresType = attentionScores.dtype();
    attentionScores = attentionScores.to(torch::kFloat32);
    torch::Tensor attention_probs = torch::softmax(attentionScores, -1);
    torch::save(attention_probs.to(at::Device(at::kCPU)), "softmax_out_example.path");
    ASD_LOG(INFO) << "softmax ok";
    // attention_probs = attention_probs.to(attentionScoresType);
    attention_probs = attention_probs.to(torch::kHalf);
    // [batch*head_num, seq_len, head_size]

    presentValue =
        presentValue.view({presentValue.sizes()[0] * presentValue.sizes()[1], presentValue.sizes()[2], presentValue.sizes()[3]});
    ASD_LOG(INFO) << "attention_probs before bmm2" << attention_probs.sizes() << " dtype:" << attention_probs.dtype();
    ASD_LOG(INFO) << "presentValue before bmm2" << presentValue.sizes() << " dtype:" << presentValue.dtype();
    // [batch*head_num, seq_len, head_size]
    torch::Tensor contextLayer = torch::bmm(attention_probs, presentValue);
    torch::save(contextLayer.to(at::Device(at::kCPU)), "bmm2_out_example.path");
    ASD_LOG(INFO) << "bmm2 ok";
    // [batch, head_num, seq_len, head_size]
    contextLayer = contextLayer.view(
        {contextLayer.sizes()[0] / param_.headNum, param_.headNum, contextLayer.sizes()[1], contextLayer.sizes()[2]});
    // [batch, seq_len, head_num, head_size]
    contextLayer = torch::transpose(contextLayer, 1, 2);
    // [batch, seq_len, head_num*head_size]
    contextLayer = contextLayer.view(
        {contextLayer.sizes()[0], contextLayer.sizes()[1], contextLayer.sizes()[2] * contextLayer.sizes()[3]});

    // [seq_len, batch, head_num*head_size]
    contextLayer = torch::transpose(contextLayer, 0, 1).contiguous();
    if (!TorchUtil::IsTensorDimEqual(contextLayer.sizes(), variantPack.outTensors[0].desc.dims)) {
        ASD_LOG(ERROR) << "contextLayer.sizes:" << contextLayer.sizes() << ", variantPack.outTensors[0].desc:"
                       << TensorUtil::AsdOpsTensorDescToString(variantPack.outTensors[0].desc);
    }
    torch::save(contextLayer.to(at::Device(at::kCPU)), "context_layer_example.path");
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, contextLayer, variantPack.outTensors[0]);

    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer