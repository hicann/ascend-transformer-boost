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
#include "chatglm6bmodel_operation.h"

namespace AclTransformer {
static const uint64_t IN_TENSOR_COUNT = 19;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

ChatGlm6BModelOperation::ChatGlm6BModelOperation(const ChatGlm6BModelParam &param)
    : GraphOperation("ChatGlm6BModelOperation"), param_(param)
{
    opGraph_.inTensorSize = IN_TENSOR_COUNT;
    opGraph_.outTensorSize = OUT_TENSOR_COUNT;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    opGraph_.nodes.resize(NODE_COUNT);
}

ChatGlm6BModelOperation::~ChatGlm6BModelOperation() {}

uint64_t ChatGlm6BModelOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm6BModelOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status ChatGlm6BModelOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                       AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer