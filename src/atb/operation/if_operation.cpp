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

Status IfOperation::GetOperationFromCondition(Operation **op) const
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

IfOperation::IfOperation(const IfCondParam &param) : OperationBase("ConditionalOperation"), param_(param) {}

IfOperation::~IfOperation()
{
    // condition为非只能指针类型时需用户手动清理

    if (param_.opA) {
        DestroyOperation(param_.opA);
    }
    if (param_.opB) {
        DestroyOperation(param_.opB);
    }
}

std::string IfOperation::GetName() const
{
    return "IfOperation";
}

Status IfOperation::Setup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context)
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(&op);
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
    st = GetOperationFromCondition(&op);
    if (st != NO_ERROR) {
        return st;
    }
    ATB_LOG(INFO) << "Calling Execute...";
    return op->Execute(variantPack, workspace, workspaceSize, context);
}

uint32_t IfOperation::GetInputNum() const
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(&op);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "Get operation from condition failed!";
        return 0;
    }
    ATB_LOG(INFO) << "Calling GetInputNum...";
    return op->GetInputNum();
}

uint32_t IfOperation::GetOutputNum() const
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(&op);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "Get operation from condition failed!";
        return 0;
    }
    ATB_LOG(INFO) << "Calling GetOutputNum...";
    return op->GetOutputNum();
}

void IfOperation::SetExecuteStreamId(uint32_t streamId)
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(&op);
    if (st != NO_ERROR) {
        return;
    }
    ATB_LOG(INFO) << "Calling SetExecuteStreamId...";
    st = atb::SetExecuteStreamId(op, streamId);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "Calling SetExecuteStreamId failed!";
        return;
    }
}

Status IfOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(&op);
    if (st != NO_ERROR) {
        return st;
    }
    ATB_LOG(INFO) << "Calling InferShape...";
    return op->InferShape(inTensorDescs, outTensorDescs);
}

std::shared_ptr<Runner> IfOperation::CreateRunner(Context &context) const
{
    Status st;
    Operation *op;
    st = GetOperationFromCondition(&op);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "Failed to get operation from condition";
        return nullptr;
    }
    OperationBase *opBase = dynamic_cast<OperationBase *>(op);
    if (!opBase) {
        ATB_LOG(ERROR) << "Failed to convert Operation to OperationBase";
        return nullptr;
    }
    ATB_LOG(INFO) << "Calling CreateRunner...";
    return opBase->CreateRunner(context);
}
} // namespace atb