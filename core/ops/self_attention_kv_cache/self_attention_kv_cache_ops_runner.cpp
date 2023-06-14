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
#include "self_attention_kv_cache_ops_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheOpsRunner::SelfAttentionKvCacheOpsRunner(const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKVCacheOperation::SelfAttentionKVCacheOperation called";
}

SelfAttentionKvCacheOpsRunner::~SelfAttentionKvCacheOpsRunner() {}

AsdOps::Status SelfAttentionKvCacheOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    // ASD_LOG(INFO) << GetName() << " SetupKernelGraph start: " << "transKey: " << param_.transKey
    //    << ",dk: " << param_.dk << ",headNum: " << param_.headNum << ",layerId: " << param_.layerId;

    // kernelGraph_.inTensors = variantPack.inTensors;
    // AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(0);
    // AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(1);
    // AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(2);
    // AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(3);
    // AsdOps::Tensor &pastKey = kernelGraph_.inTensors.at(4);
    // AsdOps::Tensor &pastValue = kernelGraph_.inTensors.at(5);

    // kernelGraph_.outTensors = variantPack.outTensors;
    // AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);
    // AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(1);
    // AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(2);

    // kernelGraph_.internalTensors.resize(8);
    // AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(0);

    // kernelGraph_.nodes.resize(8);
    // auto &mulsNode = kernelGraph_.nodes.at(0);
    // auto &permuteQNode = kernelGraph_.nodes.at(0);
    // auto &catKeyNode = kernelGraph_.nodes.at(0);
    // auto &catValueNode = kernelGraph_.nodes.at(0);
    // auto &mulsNode = kernelGraph_.nodes.at(0);
    // auto &mulsNode = kernelGraph_.nodes.at(0);
    // auto &mulsNode = kernelGraph_.nodes.at(0);
    // auto &mulsNode = kernelGraph_.nodes.at(0);
    // auto &mulsNode = kernelGraph_.nodes.at(0);
    // auto &mulsNode = kernelGraph_.nodes.at(0);

    // float varAttr = 1.0 / (sqrt(param_.dk) * (param_.layerId + 1));
    // mulsNode.opDesc = {0, "ElewiseOperation",
    //     AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    // mulsNode.inTensors = {&mixedQuery};
    // mulsNode.outTensors = {&divOut};

    // permuteQNode.opDesc = {0, "AsStridedOperation",
    //     AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    // permuteQNode.inTensors = {&mixedQuery};
    // permuteQNode.outTensors = {&divOut};
    // permuteQNode.inTensorViewFuncs = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
    // {
    //     newDims= {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    // };

    // catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    // catKeyNode.inTensors = {&mixedKey, &pastKey};
    // catKeyNode.outTensors = {&presentKey};

    // catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    // catValueNode.inTensors = {&mixedValue, &pastValue};
    // catValueNode.outTensors = {&presentValue};

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
