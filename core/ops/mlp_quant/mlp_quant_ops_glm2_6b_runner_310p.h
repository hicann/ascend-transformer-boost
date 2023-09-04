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
#ifndef MLP_QUANT_OPS_GLM2_RUNNER_310P_H
#define MLP_QUANT_OPS_GLM2_RUNNER_310P_H
#include "acltransformer/base/ops_runner.h"
#include "acltransformer/params/mlp_quant.h"

namespace AclTransformer {
class MlpQuantOpsGlm2Runner310P : public OpsRunner {
public:
    explicit MlpQuantOpsGlm2Runner310P(const MlpQuantParam &param);
    virtual ~MlpQuantOpsGlm2Runner310P();

protected:
    AsdOps::Status SetupKernelGraph(const RunnerVariantPack &runnerVariantPack) override;

private:
    AsdOps::Status SetupKernelGraphNz(const RunnerVariantPack &runnerVariantPack);
    AsdOps::Status SetupKernelGraphNd(const RunnerVariantPack &runnerVariantPack);

private:
    MlpQuantParam param_;
    AsdOps::SVector<int64_t> oriADims_;
    AsdOps::SVector<int64_t> oriBDims_;
    std::size_t oriSize_ = 3;
};
} // namespace AclTransformer
#endif