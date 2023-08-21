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
#ifndef MATMUL_OPS_RUNNER_910A_H
#define MATMUL_OPS_RUNNER_910A_H
#include "acltransformer/base/ops_runner.h"
#include "acltransformer/params/matmul.h"

namespace AclTransformer {
class MatmulOpsRunner910A : public OpsRunner {
public:
    explicit MatmulOpsRunner910A(const MatmulParam &param);
    virtual ~MatmulOpsRunner910A();

private:
    MatmulParam param_;
    AsdOps::SVector<int64_t> inputTensorDims_;
    AsdOps::SVector<int64_t> weightdims_;
};
} // namespace AclTransformer
#endif