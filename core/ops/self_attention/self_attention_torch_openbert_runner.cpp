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
#include "self_attention_torch_openbert_runner.h"
#include <cmath>
#ifdef USE_TORCH_RUNNER
#include <torch/torch.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionOpenbertTorchRunner::SelfAttentionOpenbertTorchRunner(const SelfAttentionParam &param)
    : Runner("SelfAttentionOpenbertTorchRunner"), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionOperation::SelfAttentionOperation called";
}

SelfAttentionOpenbertTorchRunner::~SelfAttentionOpenbertTorchRunner() {}

AsdOps::Status SelfAttentionOpenbertTorchRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
#ifdef USE_TORCH_RUNNER
    // 384, 32, 1024 -> 384, 32, 1024
    ASD_LOG(INFO) << "headNum:" << this->param_.headNum << "   dk:" << this->param_.dk;
    torch::Tensor mixedQuery = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[0]);
    mixedQuery = mixedQuery.view({mixedQuery.sizes()[0], mixedQuery.sizes()[1] * this->param_.headNum,
                                  mixedQuery.sizes()[2] / this->param_.headNum});
    mixedQuery = torch::transpose(mixedQuery, 0, 1);
    torch::Tensor mixedKey = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[1]);
    torch::Tensor mixedValue = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[2]);
    mixedValue = mixedValue.view({mixedValue.sizes()[0], mixedValue.sizes()[1] * this->param_.headNum,
                                  mixedValue.sizes()[2] / this->param_.headNum});
    mixedValue = torch::transpose(mixedValue, 0, 1);
    mixedKey = mixedKey.view(
        {mixedKey.sizes()[0], mixedKey.sizes()[1] * this->param_.headNum, mixedKey.sizes()[2] / this->param_.headNum});
    mixedKey = mixedKey.permute({1, 2, 0});

    torch::Tensor attention_mask = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[3]);

    double scal = 1 / sqrt(this->param_.dk);
    torch::Tensor attentionScores = torch::bmm(mixedQuery, mixedKey).contiguous();
    attentionScores = torch::mul(attentionScores, scal);
    attentionScores = attentionScores.view({attentionScores.sizes()[0] / this->param_.headNum, this->param_.headNum,
                                            attentionScores.sizes()[1], attentionScores.sizes()[2]});
    attentionScores = torch::add(attentionScores, attention_mask);
    attentionScores = attentionScores.view({attentionScores.sizes()[0] * attentionScores.sizes()[1],
                                            attentionScores.sizes()[2], attentionScores.sizes()[3]});

    torch::Tensor attention_probs = torch::softmax(attentionScores, -1);
    torch::Tensor contextLayer = torch::bmm(attention_probs, mixedValue);
    contextLayer = torch::transpose(contextLayer, 0, 1).contiguous();
    torch::Tensor atOutTensor = contextLayer
                                    .view({contextLayer.sizes()[0], contextLayer.sizes()[1] / this->param_.headNum,
                                           contextLayer.sizes()[2] * this->param_.headNum})
                                    .contiguous();

    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, runnerVariantPack.outTensors[0]);

    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer