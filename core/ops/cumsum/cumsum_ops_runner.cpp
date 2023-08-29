

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 *  * Licensed under the Apache License, Version 2.0 (the "License");
 *  * you may not use this file except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 * http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  */
#include "cumsum_ops_runner.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/params/cumsum.h"

namespace AclTransformer {
CumsumOpsRunner::CumsumOpsRunner(const CumsumParam &param) : OpsRunner("CumsumOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "CumsumOpsRunner::CumsumOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &cumsumNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Cumsum asdParam;
    asdParam.exclusive = param_.exclusive;
    asdParam.reverse = param_.reverse;
    for (std::size_t i = 0; i < param_.axes.size(); ++i) {
        asdParam.axis.push_back(param_.axes[i]);
    }

    cumsumNode.opDesc = {0, "CumsumOperation", asdParam};
    cumsumNode.inTensors = {&xTensor};
    cumsumNode.outTensors = {&resultTensor};
}

CumsumOpsRunner::~CumsumOpsRunner() {}

} // namespace AclTransformer
