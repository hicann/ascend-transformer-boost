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
#ifndef ACLTRANSFORMER_POSITION_EMBEDDING_FUSION_OPERATION_H
#define ACLTRANSFORMER_POSITION_EMBEDDING_FUSION_OPERATION_H
#include "acltransformer/operation.h"
#include "acltransformer/params/position_embedding_fusion.h"

namespace AclTransformer {
class OptRopeOperation : public Operation {
public:
    OptRopeOperation(const PositionEmbeddingFusionParam &param);
    virtual ~OptRopeOperation();
    uint64_t GetInTensorCount() const override;
    uint64_t GetOutTensorCount() const override;

protected:
    AsdOps::Status InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;
    RunnerBuilder *FindBestRunnerBuilder() const override;

private:
    PositionEmbeddingFusionParam param_;
    const size_t IN_TENSOR_SIZE = 4;
    const size_t OUT_TENSOR_SIZE = 3;
    const int32_t KQV_SLICE_SIZE = 3;
};
} // namespace AclTransformer
#endif