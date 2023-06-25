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
#include "position_embedding_1d_split_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbedding1dSplitOpsRunner::PositionEmbedding1dSplitOpsRunner(const PositionEmbedding1dSplitParam &param)
    : OpsRunner("PositionEmbedding1dSplitOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_1D_SPLIT), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbedding1dSplitOpsRunner::PositionEmbedding1dSplitOpsRunner called";
}

PositionEmbedding1dSplitOpsRunner::~PositionEmbedding1dSplitOpsRunner() {}

AsdOps::Status PositionEmbedding1dSplitOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " SetupKernelGraph start: "
                  << "headNum: " << param_.headNum;

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
