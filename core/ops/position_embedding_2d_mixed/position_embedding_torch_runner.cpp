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
#include "position_embedding_torch_runner.h"
#include <cmath>
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingTorchRunner::PositionEmbeddingTorchRunner(const PositionEmbeddingParam &param)
    : Runner("PositionEmbeddingTorchRunner"), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingTorchRunner::PositionEmbeddingTorchRunner called";
}

PositionEmbeddingTorchRunner::~PositionEmbeddingTorchRunner() {}

AsdOps::Status PositionEmbeddingTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
#ifdef USE_TORCH_RUNNER
    // in : [seq_len, batch, all_head_size]   position_ids,[]  cos_table,[]  sin_table[]
    // out : [seq_len, batch, head_num, head_size]
    ASD_LOG(INFO) << "headNum:" << this->param_.headNum;
    torch::Tensor mixed = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[0]);
    torch::Tensor positionIds = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[1]);
    torch::Tensor cosTable = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[2]);
    torch::Tensor sinTable = TorchUtil::AsdOpsTensor2AtTensor(handle, variantPack.inTensors[3]);
    ASD_LOG(INFO) << "start";
    ASD_LOG(INFO) << "mixed" << mixed.sizes();
    ASD_LOG(INFO) << "positionIds" << positionIds.sizes();
    ASD_LOG(INFO) << "cosTable" << cosTable.sizes();
    ASD_LOG(INFO) << "sinTable" << sinTable.sizes();

    mixed =
        mixed.view({mixed.sizes()[0], mixed.sizes()[1], this->param_.headNum, mixed.sizes()[2] / this->param_.headNum});
    std::vector<torch::Tensor> qkvLayer = mixed.chunk(3, -1);
    std::vector<torch::Tensor> qChunks = qkvLayer[0].chunk(2, -1);
    std::vector<torch::Tensor> kChunks = qkvLayer[1].chunk(2, -1);

    torch::Tensor positionIds1 = positionIds.slice(1, 0, 1).squeeze(1).transpose(0, 1).contiguous();
    torch::Tensor positionIds2 = positionIds.slice(1, 1, 2).squeeze(1).transpose(0, 1).contiguous();
    ASD_LOG(INFO) << "split positionIds" << positionIds1.sizes() << " " << positionIds2.sizes();

    torch::Tensor cos1 = torch::nn::functional::embedding(positionIds1, cosTable.squeeze(1)).unsqueeze(2);
    torch::Tensor sin1 = torch::nn::functional::embedding(positionIds1, sinTable.squeeze(1)).unsqueeze(2);
    torch::Tensor cos2 = torch::nn::functional::embedding(positionIds2, cosTable.squeeze(1)).unsqueeze(2);
    torch::Tensor sin2 = torch::nn::functional::embedding(positionIds2, sinTable.squeeze(1)).unsqueeze(2);
    ASD_LOG(INFO) << "cos, sin " << cos1.sizes() << " " << sin1.sizes();

    int chunksLastDim = qChunks[0].sizes().size() - 1;
    int chunksLastDimSize = qChunks[0].sizes()[chunksLastDim];
    ASD_LOG(INFO) << "chunksLastDim: " << chunksLastDim;
    ASD_LOG(INFO) << "chunksLastDimSize: " << chunksLastDimSize;
    torch::Tensor qRotate1 = torch::cat({qChunks[0].slice(-1, chunksLastDimSize / 2, chunksLastDimSize).neg(),
                                         qChunks[0].slice(-1, 0, chunksLastDimSize / 2)},
                                        chunksLastDim);
    torch::Tensor qRotate2 = torch::cat({qChunks[1].slice(-1, chunksLastDimSize / 2, chunksLastDimSize).neg(),
                                         qChunks[1].slice(-1, 0, chunksLastDimSize / 2)},
                                        chunksLastDim);
    torch::Tensor kRotate1 = torch::cat({kChunks[0].slice(-1, chunksLastDimSize / 2, chunksLastDimSize).neg(),
                                         kChunks[0].slice(-1, 0, chunksLastDimSize / 2)},
                                        chunksLastDim);
    torch::Tensor kRotate2 = torch::cat({kChunks[1].slice(-1, chunksLastDimSize / 2, chunksLastDimSize).neg(),
                                         kChunks[1].slice(-1, 0, chunksLastDimSize / 2)},
                                        chunksLastDim);
    ASD_LOG(INFO) << "qRotate1: " << qRotate1.sizes();
    torch::Tensor qEmbedded1 = torch::add(torch::mul(qChunks[0], cos1), torch::mul(qRotate1, sin1));
    torch::Tensor qEmbedded2 = torch::add(torch::mul(qChunks[1], cos2), torch::mul(qRotate2, sin2));
    torch::Tensor kEmbedded1 = torch::add(torch::mul(kChunks[0], cos1), torch::mul(kRotate1, sin1));
    torch::Tensor kEmbedded2 = torch::add(torch::mul(kChunks[1], cos2), torch::mul(kRotate2, sin2));

    torch::Tensor qEmbedded = torch::cat({qEmbedded1, qEmbedded2}, chunksLastDim).contiguous();
    torch::Tensor kEmbedded = torch::cat({kEmbedded1, kEmbedded2}, chunksLastDim).contiguous();
    ASD_LOG(INFO) << "qEmbedded: " << qEmbedded.sizes() << ", variantPack.outTensors[0].desc:"
                  << TensorUtil::AsdOpsTensorDescToString(variantPack.outTensors[0].desc);
    ASD_LOG(INFO) << "kEmbedded: " << kEmbedded.sizes() << ", variantPack.outTensors[0].desc:"
                  << TensorUtil::AsdOpsTensorDescToString(variantPack.outTensors[0].desc);

    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, qEmbedded.contiguous(), variantPack.outTensors[0]);
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, kEmbedded.contiguous(), variantPack.outTensors[1]);
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, qkvLayer[2].contiguous(), variantPack.outTensors[2]);

    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer