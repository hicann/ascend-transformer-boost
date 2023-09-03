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
#ifndef SELFATTENTIONCROSS_OPS_LLAMA7BADAPTER_RUNNER_ADAPTER_310P_H
#define SELFATTENTIONCROSS_OPS_LLAMA7BADAPTER_RUNNER_ADAPTER_310P_H
#include "acltransformer/base/ops_runner.h"
#include "acltransformer/params/self_attention_cross.h"

namespace AclTransformer {
class SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p : public OpsRunner {
public:
    SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p(const SelfAttentionCrossParam &param);
    virtual ~SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p();

private:
    void AsStrideKernelInferShapeSet(const AsdOps::SVector<int64_t> &sequence, KernelGraphNode &node);

private:
    SelfAttentionCrossParam param_;
    AsdOps::SVector<int64_t> orgQDims_;
    AsdOps::SVector<int64_t> orgKDims_;
    AsdOps::SVector<int64_t> orgProbsDims_;
    AsdOps::SVector<int64_t> orgVDims_;
};

} // namespace AclTransformer
#endif