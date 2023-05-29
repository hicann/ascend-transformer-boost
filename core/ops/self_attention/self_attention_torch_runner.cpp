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
#include "self_attention_torch_runner.h"
#include "acltransformer/utils/tensor_util.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <ATen/ATen.h>
#include <cmath>

namespace AclTransformer {
SelfAttentionTorchRunner::SelfAttentionTorchRunner(const SelfAttentionParam &param)
    : Runner("SelfAttentionTorchRunner"), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionOperation::SelfAttentionOperation called";
}

SelfAttentionTorchRunner::~SelfAttentionTorchRunner() {}

AsdOps::Status SelfAttentionTorchRunner::Init() { return AsdOps::Status::OkStatus(); }

AsdOps::Status SelfAttentionTorchRunner::Setup(Handle &handle, VariantPack &runInfo)
{
    return AsdOps::Status::OkStatus();
}

uint64_t SelfAttentionTorchRunner::GetWorkspaceSize() { return 0; }

AsdOps::Status SelfAttentionTorchRunner::Execute(Handle &handle, VariantPack &runInfo)
{
    // 384, 32, 1024 -> 384, 32, 1024
    ASD_LOG(INFO) << "headNum:" << this->param_.headNum << "   dk:" << this->param_.dk;
    torch::Tensor mixedQuery = AsdOpsTensor2AtTensor(runInfo.inTensors[0]);
    mixedQuery = mixedQuery.view({mixedQuery.sizes()[0], mixedQuery.sizes()[1] * this->param_.headNum,
                                  mixedQuery.sizes()[2] / this->param_.headNum});
    mixedQuery = torch::transpose(mixedQuery, 0, 1);
    torch::Tensor mixedKey = AsdOpsTensor2AtTensor(runInfo.inTensors[1]);
    torch::Tensor mixedValue = AsdOpsTensor2AtTensor(runInfo.inTensors[2]);
    mixedValue = mixedValue.view({mixedValue.sizes()[0], mixedValue.sizes()[1] * this->param_.headNum,
                                  mixedValue.sizes()[2] / this->param_.headNum});
    mixedValue = torch::transpose(mixedValue, 0, 1);
    mixedKey = mixedKey.view(
        {mixedKey.sizes()[0], mixedKey.sizes()[1] * this->param_.headNum, mixedKey.sizes()[2] / this->param_.headNum});
    mixedKey = mixedKey.permute({1, 2, 0});

    torch::Tensor attention_mask = AsdOpsTensor2AtTensor(runInfo.inTensors[3]);

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
    contextLayer = contextLayer
                       .view({contextLayer.sizes()[0], contextLayer.sizes()[1] / this->param_.headNum,
                              contextLayer.sizes()[2] * this->param_.headNum})
                       .contiguous();

    int ret = AsdRtMemCopyAsync(runInfo.outTensors[0].data, runInfo.outTensors[0].dataSize,
                                contextLayer.storage().data_ptr().get(), runInfo.outTensors[0].dataSize,
                                ASDRT_MEMCOPY_DEVICE_TO_DEVICE, handle.stream);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtMemCopy fail";
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer