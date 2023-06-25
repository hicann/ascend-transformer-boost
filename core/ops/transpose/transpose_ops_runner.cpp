

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
#include "transpose_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
TransposeOpsRunner::TransposeOpsRunner(const TransposeParam &param)
    : OpsRunner("TransposeOpsRunner", RUNNER_TYPE_TRANSPOSE), param_(param)
{
    ASD_LOG(INFO) << "TransposeOpsRunner::TransposeOpsRunner called, param_.dimA:" << param_.dimA
                  << " param_.dimB:" << param_.dimB;
    kernelGraph_.inTensors.resize(1);
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &operationInTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

    size_t size = operationInTensor.desc.dims.size();
    AsdOps::SVector<int64_t> sizeParam = operationInTensor.desc.dims;
    AsdOps::SVector<int64_t> strideParam(size);
    AsdOps::SVector<int64_t> offsetParam = {0};
    int64_t stride = 1;
    for (size_t i = 0; i < size; i++) {
        strideParam.at(size - i - 1) = stride;
        stride *= sizeParam.at(size - i - 1);
    }
    std::swap(sizeParam[param_.dimA], sizeParam[param_.dimB]);
    std::swap(strideParam[param_.dimA], strideParam[param_.dimB]);

    kernelGraph_.nodes.resize(1);
    auto &asstridedNode = kernelGraph_.nodes.at(0);
    asstridedNode.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided({sizeParam, strideParam, offsetParam})};
    asstridedNode.inTensors = {&operationInTensor};
    asstridedNode.outTensors = {&operationOutTensor};
}

TransposeOpsRunner::~TransposeOpsRunner() {}
} // namespace AclTransformer
