/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file ex10cept in compliance with the License.
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
#ifndef LINEAR_OPS_RUNNER_H
#define LINEAR_OPS_RUNNER_H
#include "acltransformer/base/ops_runner.h"
#include "acltransformer/params/linear.h"

namespace AclTransformer {
class LinearOpsRunner : public OpsRunner {
public:
    LinearOpsRunner(LinearParam &param);
    virtual ~LinearOpsRunner();

protected:
    AsdOps::Status SetupKernelGraph(const VariantPack &variantPack) override;
    AsdOps::Status ExecuteImpl(Handle &handle, VariantPack &variantPack) override;

private:
    AsdOps::Status SetupKernelGraph910A(const VariantPack &variantPack);
    AsdOps::Status SetupKernelGraph910B(const VariantPack &variantPack);
    void ConvertNewVariantPack(const VariantPack &variantPack, VariantPack &newVariantPack,
                               AsdOps::SVector<int64_t> &matmulOrgShape, AsdOps::SVector<int64_t> &transdataOrgShape);

private:
    LinearParam param_;
    bool is910B_ = false;
};

} // namespace AclTransformer
#endif