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
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"
#include "transdata_int8_ops_runner.h"

namespace AclTransformer {
TransDataInt8OpsRunner::TransDataInt8OpsRunner(const TransDataInt8Param &param)
    : OpsRunner("TransDataInt8OpsRunner", RUNNER_TYPE_TRANSDATA_INT8), param_(param)
{
    const std::size_t inTensorCount = 1;

    ASD_LOG(INFO) << "TransDataInt8OpsRunner::TransDataInt8OpsRunner called";
    kernelGraph_.inTensors.resize(inTensorCount);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors[0];

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &yTensor = kernelGraph_.outTensors[0];

    kernelGraph_.nodes.resize(1);
    auto &transdata1Node = kernelGraph_.nodes[0];

    ViewFunc CheckDimB = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimB_ = oldDims;
        newDims = {1, oldDims.at(0), oldDims.at(1)};
    };
    transdata1Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata1Node.inTensors = {&xTensor};
    transdata1Node.outTensors = {&yTensor};
    transdata1Node.inTensorViewFuncs.resize(transdata1Node.inTensors.size());
    transdata1Node.inTensorViewFuncs.at(0) = CheckDimB;
}

TransDataInt8OpsRunner::~TransDataInt8OpsRunner() {}
} // namespace AclTransformer
