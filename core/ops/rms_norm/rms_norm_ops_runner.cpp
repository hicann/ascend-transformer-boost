

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
#include "rms_norm_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
RmsNormOpsRunner::RmsNormOpsRunner(const RmsNormParam &param)
    : OpsRunner("RmsNormOpsRunner", RUNNER_TYPE_RMS_NORM), param_(param)
{
}

RmsNormOpsRunner::~RmsNormOpsRunner() {}

AsdOps::Status RmsNormOpsRunner::SetupKernelGraph(const VariantPack &variantPack) { return AsdOps::Status::OkStatus(); }

bool RmsNormOpsRunner::CalcLayerNormTensor(const VariantPack &variantPack, int64_t &beginDim) { return true; }
} // namespace AclTransformer
