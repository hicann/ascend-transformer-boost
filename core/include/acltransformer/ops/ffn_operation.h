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
#ifndef ACLTRANSFORMER_OPS_FFN_OPERATION_H
#define ACLTRANSFORMER_OPS_FFN_OPERATION_H
#include "acltransformer/operation.h"
#include "acltransformer/params/ffn.h"

namespace AclTransformer {
// gelu(A*B+Bias)
class FfnOperation : public Operation {
public:
    FfnOperation(const FfnParam &param);
    virtual ~FfnOperation();
    AsdOps::Status InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) override;
    bool IsConsistent(const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs,
                      AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const;
    int64_t GetTensorBatch(const AsdOps::TensorDesc &tensorDesc) const;
    int64_t GetTensorH(const AsdOps::TensorDesc &tensorDesc) const;
    int64_t GetTensorW(const AsdOps::TensorDesc &tensorDesc) const;

protected:
    RunnerBuilder *FindBestRunnerBuilder() override;

private:
    FfnParam param_;
};
} // namespace AclTransformer
#endif