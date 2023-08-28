

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 *  * Licensed under the Apache License, Version 2.0 (the "License");
 *  * you may not use this file except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 * http://www.apache.org/licenses/LICENSE-2.0
 *  *
 * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  */
#include "sort_ops_runner.h"
#include "acltransformer/params/sort.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
SortOpsRunner::SortOpsRunner(const SortParam &param) : OpsRunner("SortOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "SortOpsRunner::SortOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(2);
    AsdOps::Tensor &resultTensor0 = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &resultTensor1 = kernelGraph_.outTensors.at(1);

    kernelGraph_.nodes.resize(1);
    auto &sortNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Sort asdParam;
    for (std::size_t i = 0; i < param_.num.size(); ++i) {
        asdParam.num.push_back(param_.num[i]);
    }

    sortNode.opDesc = {0, "SortOperation", asdParam};
    sortNode.inTensors = {&xTensor};
    sortNode.outTensors = {&resultTensor0, &resultTensor1};
}

SortOpsRunner::~SortOpsRunner() {}

} // namespace AclTransformer
