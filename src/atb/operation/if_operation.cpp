/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation/if_operation.h"
#include "atb/types.h"
#include "atb/utils/log.h"
#include "atb/operation/plugin_operation.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/common_utils.h"

namespace atb {
const size_t MAX_NODE_NUM = 1024;
const size_t MAX_GRAPH_NAME_LEN = 128;

Status IfOperation::GetOperationFromCondition(const void *condition, Operation **op)
{
    bool cond;
    try {
        cond = param_.handle(param_.condition);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << "Get condition failed, please check handle function";
        return ERROR_INVALID_PARAM;
    }

    if (cond && param_.opA) {
        ATB_LOG(INFO) << "Condition met (true), selecting opA...";
        *op = param_.opA;
    } else if (!cond && param_.opB) {
        ATB_LOG(INFO) << "Condition not met (false), selecting opB...";
        *op = param_.opB;
    } else {
        ATB_LOG(ERROR) << "Please check the intended operation is valid, opA: " << param_.opA << " opB: " << param_.opB;
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

template <> Status CreateOperation(const IfCondParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        ATB_LOG(ERROR) << "Invalid param, operation is nullptr";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) IfOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "Failed to new conditional operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

IfOperation::IfOperation(const IfCondParam &param) : OperationBase("ConditionalOperation"), param_(param)
{
    if (!param_.opA && !param_.opB) {
        return 
    }
}

IfOperation::~IfOperation()
{
    // TODO: any cleanup if necessary
}

Status IfOperation::Setup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context)
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(param_.condition, &op);
    if (st != NO_ERROR) {
        return st;
    }
    ATB_LOG(INFO) << "Calling Setup...";
    return op->Setup(variantPack, workspaceSize, context);
}

Status IfOperation::Execute(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                            Context *context)
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(param_.condition, &op);
    if (st != NO_ERROR) {
        return st;
    }
    ATB_LOG(INFO) << "Calling Execute...";
    return op->Execute(variantPack, workspace, workspaceSize, context);
}

uint32_t IfOperation::GetInputNum() const
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Getting input num...";
    return op->GetInputNum();
}

uint32_t IfOperation::GetOutputNum() const
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Getting output num...";
    return op->GetOutputNum();
}

void SetExecuteStreamId(uint32_t streamId)
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Setting streamId...";
    op->SetExecuteStreamId(streamId);
}

Status IfOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(param_.condition, &op);
    if (st != NO_ERROR) {
        return st;
    }
    ATB_LOG(INFO) << "Calling InferShapeImpl...";
    return op->InferShapeImpl(inTensorDescs, outTensorDescs);
}

std::shared_ptr<Runner> IfOperation::CreateRunner(Context &context) const
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Calling CreateRunner...";
    return op->CreateRunner(context);
}

void IfOperation::InitEmptyInTensorPerms()
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Calling InitEmptyInTensorPerms...";
    op->InitEmptyInTensorPerms();
}

SVector<bool> IfOperation::GetEmptyInTensorPermissions() const
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Calling GetEmptyInTensorPermissions...";
    return op->GetEmptyInTensorPermissions();
}

void IfOperation::InitEmptyOutTensorPerms()
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Calling InitEmptyOutTensorPerms...";
    op->InitEmptyOutTensorPerms();
}

SVector<bool> IfOperation::GetEmptyOutTensorPermissions() const
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Calling GetEmptyOutTensorPermissions...";
    return op->GetEmptyOutTensorPermissions();
}

void IfOperation::GetGraphInfoImpl(nlohmann::json &graphJson) const
{
    Operation *op;
    GetOperationFromCondition(param_.condition, &op);
    ATB_LOG(INFO) << "Calling GetGraphInfoImpl...";
    nlohmann::json opGraphJson;
    op->GetGraphInfoImpl(opGraphJson);
}

} // namespace atb