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
#include "position_embedding_1d_split_torch_runner.h"
#include <cmath>
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbedding1dSplitTorchRunner::PositionEmbedding1dSplitTorchRunner(const PositionEmbedding1dSplitParam &param)
    : Runner("PositionEmbedding1dSplitTorchRunner"), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbedding1dSplitOperation::PositionEmbedding1dSplitOperation called";
}

PositionEmbedding1dSplitTorchRunner::~PositionEmbedding1dSplitTorchRunner() {}

AsdOps::Status PositionEmbedding1dSplitTorchRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
#ifdef USE_TORCH_RUNNER
    // in : Q,[batch, seq_len, all_head_size]   position_ids,[]  cos_table,[]  sin_table[]
    // out : Q ,[seq_len, batch, head_num, head_size]
    ASD_LOG(INFO) << "headNum:" << this->param_.headNum;
    torch::Tensor input = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[0]);
    torch::Tensor positionIds = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[1]);
    torch::Tensor cosTable = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[2]);
    torch::Tensor sinTable = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[3]);
    // [batch, headNum, seqlen, headDim]
    input = input.view({input.sizes()[0], input.sizes()[1], param_.headNum, input.sizes()[2] / param_.headNum})
                .transpose(1, 2);
    // [seqLen, head_dim]
    cosTable = cosTable.squeeze(1).squeeze(0);
    sinTable = sinTable.squeeze(1).squeeze(0);
    // [bs, 1, seqlen, headDim]
    torch::Tensor cos = torch::nn::functional::embedding(positionIds, cosTable).unsqueeze(1);
    torch::Tensor sin = torch::nn::functional::embedding(positionIds, sinTable).unsqueeze(1);
    int chunksLastDim = input.sizes().size() - 1;
    int chunksLastDimSize = input.sizes()[chunksLastDim];
    ASD_LOG(INFO) << "chunksLastDim: " << chunksLastDim;
    ASD_LOG(INFO) << "chunksLastDimSize: " << chunksLastDimSize;
    torch::Tensor inputRotate = torch::cat(
        {input.slice(-1, chunksLastDimSize / 2, chunksLastDimSize).neg(), input.slice(-1, 0, chunksLastDimSize / 2)},
        chunksLastDim);
    // [batch, headNum, seqlen, headDim]
    torch::Tensor inputEmbedded = torch::add(torch::mul(input, cos), torch::mul(inputRotate, sin));
    // [seqlen, batch, headNum, headDim]
    inputEmbedded = inputEmbedded.permute({2, 0, 1, 3});
    ASD_LOG(INFO) << "inputEmbedded: " << inputEmbedded.sizes();
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, inputEmbedded.contiguous(), runnerVariantPack.outTensors[0]);
    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer