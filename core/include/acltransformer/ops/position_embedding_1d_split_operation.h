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
#ifndef ACLTRANSFORMER_POSITION_EMBEDDING_1D_SPLIT_OPERATION_H
#define ACLTRANSFORMER_POSITION_EMBEDDING_1D_SPLIT_OPERATION_H
#include "acltransformer/operation.h"
#include "acltransformer/params/position_embedding_1d_split.h"

namespace AclTransformer {
class PositionEmbedding1dSplitOperation : public Operation {
public:
    PositionEmbedding1dSplitOperation(const PositionEmbedding1dSplitParam &param);
    virtual ~PositionEmbedding1dSplitOperation();
    AsdOps::Status InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) override;

protected:
    RunnerBuilder *FindBestRunnerBuilder() override;

private:
    PositionEmbedding1dSplitParam param_;
    Runner *runner_ = nullptr;
};
} // namespace AclTransformer
#endif