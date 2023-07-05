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
#include "chatglm6b28_operation.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
ChatGlm6B28Operation::ChatGlm6B28Operation(ChatGlm6B28Param &param)
    : GraphOperation("ChatGlm6B28Operation"), param_(param)
{
    BuildGraph();
}

ChatGlm6B28Operation::~ChatGlm6B28Operation() {}

uint64_t ChatGlm6B28Operation::GetInTensorCount() const { return 0; }

uint64_t ChatGlm6B28Operation::GetOutTensorCount() const { return 0; }

AsdOps::Status ChatGlm6B28Operation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                    AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    return AsdOps::Status::OkStatus();
}

void ChatGlm6B28Operation::BuildGraph() {}
} // namespace AclTransformer
