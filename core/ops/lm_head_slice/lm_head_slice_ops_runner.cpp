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
#include "lm_head_slice_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
enum InTensorId {
    IN_INPUT = 0,
    IN_TENSOR_MAX,
};

static const uint64_t NODE_COUNT = 1;

LmHeadSliceOpsRunner::LmHeadSliceOpsRunner(const LmHeadSliceParam &param) : OpsRunner("LmHeadSliceOpsRunner", RUNNER_TYPE_LMHEAD_SLICE), param_(param)
{
    ASD_LOG(INFO) << "LmHeadSliceOpsRunner::LmHeadSliceOpsRunner called";

    kernelGraph_.nodes.resize(NODE_COUNT);
    int64_t nodeId = 0;

    kernelGraph_.inTensors.resize(IN_TENSOR_MAX);
    kernelGraph_.outTensors.resize(1);

    auto &sliceNode = kernelGraph_.nodes.at(nodeId++);
    sliceNode.opDesc = {0, "SliceOperation",
                            AsdOps::OpParam::Slice{AsdOps::OpParam::Slice::SLICE, {param_.seqLen - 1, 0, 0}, {1, -1, -1}}};
    sliceNode.inTensors = {&kernelGraph_.inTensors.at(IN_INPUT)};
    sliceNode.outTensors = {&kernelGraph_.outTensors.at(0)};
}

LmHeadSliceOpsRunner::~LmHeadSliceOpsRunner() {}
} // namespace AclTransformer
