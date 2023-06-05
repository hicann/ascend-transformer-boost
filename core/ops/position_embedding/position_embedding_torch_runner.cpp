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
#include "position_embedding_torch_runner.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/utils/tensor_cache.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <ATen/ATen.h>
#include <cmath>

namespace AclTransformer {
PositionEmbeddingTorchRunner::PositionEmbeddingTorchRunner(const PositionEmbeddingParam &param)
    : Runner("PositionEmbeddingTorchRunner"), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingOperation::PositionEmbeddingOperation called";
}

PositionEmbeddingTorchRunner::~PositionEmbeddingTorchRunner() {}

AsdOps::Status PositionEmbeddingTorchRunner::ExecuteImpl(Handle &handle, VariantPack &variantPack)
{
    // in : mixed,[seq_len, batch, all_head_size]   position_ids,[]  cos_table,[]  sin_table[]
    // out : mixed ,[seq_len, batch, head_num, head_size]
    ASD_LOG(INFO) << "headNum:" << this->param_.headNum;
    torch::Tensor mixed = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[0].data);
    torch::Tensor positionIds = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[1].data);
    torch::Tensor cosTable = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[2].data);
    torch::Tensor sinTable = *AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.inTensors[3].data);
    ASD_LOG(INFO) << "start";
    ASD_LOG(INFO) << "mixed" << mixed.sizes();
    ASD_LOG(INFO) << "positionIds" << positionIds.sizes();
    ASD_LOG(INFO) << "cosTable" << cosTable.sizes();
    ASD_LOG(INFO) << "sinTable" << sinTable.sizes();

    mixed = mixed.view({mixed.sizes()[0], mixed.sizes()[1], this->param_.headNum, mixed.sizes()[2] / this->param_.headNum});
    std::vector<torch::Tensor> chunks = mixed.chunk(2, -1);
    ASD_LOG(INFO) << "split mixed" << chunks.at(0).sizes() << " " << chunks.at(1).sizes();

    torch::Tensor positionIds1 = positionIds.slice(1, 0, 1).squeeze(1).transpose(0, 1).contiguous();
    torch::Tensor positionIds2 = positionIds.slice(1, 1, 2).squeeze(1).transpose(0, 1).contiguous();
    ASD_LOG(INFO) << "split positionIds" << positionIds1.sizes() << " " << positionIds2.sizes();

    torch::Tensor cos = torch::nn::functional::embedding(positionIds1, cosTable.squeeze(1)).unsqueeze(2);
    torch::Tensor sin = torch::nn::functional::embedding(positionIds2, sinTable.squeeze(1)).unsqueeze(2);
    ASD_LOG(INFO) << "cos, sin " << cos.sizes() << " " << sin.sizes();

    torch::Tensor rotate1 = torch::cat({chunks[0].slice(-1, chunks[0].sizes()[-1] / 2, chunks[0].sizes()[-1]).neg(), 
                                      chunks[0].slice(-1, 0, chunks[0].sizes()[-1] / 2)}, - 1);
    torch::Tensor rotate2 = torch::cat({chunks[1].slice(-1, chunks[1].sizes()[-1] / 2, chunks[1].sizes()[-1]).neg(), 
                                      chunks[1].slice(-1, 0, chunks[1].sizes()[-1] / 2)}, - 1);

    torch::Tensor embedded1 = torch::add(torch::mul(chunks[0], cos), torch::mul(rotate1, sin));
    torch::Tensor embedded2 = torch::add(torch::mul(chunks[1], cos), torch::mul(rotate2, sin));

    torch::Tensor embedded = torch::cat({embedded1, embedded2}, -1);
    

    torch::Tensor *atOutTensor = AsdOps::GetSingleton<TensorCache>().GetTensor(variantPack.outTensors[0].data);
    *atOutTensor = embedded;

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer