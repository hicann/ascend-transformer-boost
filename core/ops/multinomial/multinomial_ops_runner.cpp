

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
#include "multinomial_ops_runner.h"
#include "acltransformer/params/multinomial.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
MultinomialOpsRunner::MultinomialOpsRunner(const MultinomialParam &param)
    : OpsRunner("MultinomialOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "MultinomialOpsRunner::MultinomialOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &multiNomialNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Multinomial asdParam;
    asdParam.multinomialType = AsdOps::OpParam::Multinomial::MULTINOMIAL;
    asdParam.numSamples = param_.numSamples;

    multiNomialNode.opDesc = {0, "MultinomialOperation", asdParam};
    multiNomialNode.inTensors = {&xTensor};
    multiNomialNode.outTensors = {&resultTensor};
}

MultinomialOpsRunner::~MultinomialOpsRunner() {}

} // namespace AclTransformer
