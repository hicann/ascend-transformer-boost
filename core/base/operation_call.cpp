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
#include "acltransformer/operation_call.h"
#include <functional>
#include <map>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/operation.h"
#include "acltransformer/plan.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
using OperationCreateFunc = std::function<Operation *(const AsdOps::Any &opParam)>;

Operation *AddCreate(const AsdOps::Any &opParam)
{
    const auto &param = AsdOps::AnyCast<AddParam>(opParam);
    return new AddOperation(param);
}

Operation *LinearCreate(const AsdOps::Any &opParam)
{
    const auto &param = AsdOps::AnyCast<LinearParam>(opParam);
    return new LinearOperation(param);
}

Operation *AddNormCreate(const AsdOps::Any &opParam)
{
    const auto &param = AsdOps::AnyCast<AddNormParam>(opParam);
    return new AddNormOperation(param);
}

Operation *FfnCreate(const AsdOps::Any &opParam)
{
    const auto &param = AsdOps::AnyCast<FfnParam>(opParam);
    return new FfnOperation(param);
}

std::map<std::string, OperationCreateFunc> OPERATION_CREATE_FUNC_MAP = {
    {"AddOperation", &AddCreate},
    {"LinearOperation", &LinearCreate},
    {"AddNormOperation", &AddNormCreate},
    {"FfnOperation", &FfnCreate},
};

OperationCall::OperationCall(const std::string &opName, const AsdOps::Any &opParam)
{
    auto it = OPERATION_CREATE_FUNC_MAP.find(opName);
    if (it == OPERATION_CREATE_FUNC_MAP.end()) {
        ASD_LOG(ERROR) << "not support operation:" << opName;
        return;
    }

    try {
        Operation *operation = it->second(opParam);
        operation_.reset(operation);
    } catch (...) {
        ASD_LOG(ERROR) << "invalid opName" << opName << " opParam";
        return;
    }

    plan_ = std::make_shared<Plan>();
    if (!plan_) {
        ASD_LOG(ERROR) << "new Plan fail";
        return;
    }

    AsdOps::Status st = operation_->BuildPlan(plan_.get());
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "build plan fail, error:" << st.Message();
        plan_.reset();
        operation_.reset();
    }
}

OperationCall::~OperationCall() {}

int OperationCall::ExecuteSync(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                               const AsdOps::SVector<AsdOps::Tensor> &outTensors, void *stream)
{
    if (!plan_) {
        return 1;
    }

    VariantPack variantPack;
    variantPack.inTensors = inTensors;
    variantPack.outTensors = outTensors;

    ASD_LOG(INFO) << "OperationCall::ExecuteSync start stream:" << stream << ", variantPack:\n"
                  << variantPack.ToString();

    Handle handle = {stream};
    AsdOps::Status st = plan_->Setup(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "plan setup fail, error:" << st.Message();
        return 1;
    }

    variantPack.workspaceSize = plan_->GetWorkspaceSize();
    ASD_LOG(INFO) << "variantPack.workspaceSize:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        int st = AsdRtMemMallocDevice(&variantPack.workspace, variantPack.workspaceSize, ASDRT_MEM_DEFAULT);
        if (st != ASDRT_SUCCESS) {
            ASD_LOG(ERROR) << "AsdRtMemMallocDevice fail, ret:" << st;
            return 1;
        }
    }

    st = plan_->Execute(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "OperationCall::ExecuteSync fail, for Plan Execute fail, error:" << st.Message();
        AsdRtMemFreeDevice(variantPack.workspace);
        return 1;
    }

    int ret = AsdRtStreamSynchronize(handle.stream);
    ASD_LOG_IF(ret != ASDRT_SUCCESS, ERROR) << "AsdRtStreamSynchronize fail, error:" << ret;

    ret = AsdRtMemFreeDevice(variantPack.workspace);
    ASD_LOG_IF(ret != ASDRT_SUCCESS, ERROR) << "AsdRtMemFreeDevice fail, error:" << ret;

    ASD_LOG(INFO) << "OperationCall::ExecuteSync success";
    return 0;
}

int OperationCall::ExecuteAsync(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                const AsdOps::SVector<AsdOps::Tensor> &outTensors, void *stream)
{
    return 0;
}
} // namespace AclTransformer