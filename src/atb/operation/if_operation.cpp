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

Status IfOperation::GetOperationFromCondition(Operation **op)
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

IfOperation::IfOperation(const IfCondParam &param) : OperationBase("IfOperation"), param_(param)
{
    if (!opSelected_) {
        ATB_LOG(INFO) << "Operation not selected yet, setting opSelected_...";
        Status st;
        st = GetOperationFromCondition(&opSelected_);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << "Failed to select operation based on condition!";
        }
    }
}

IfOperation::~IfOperation()
{
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
    ATB_LOG(INFO) << "Calling Setup...";
    return opSelected_->Setup(variantPack, workspaceSize, context);
}

Status IfOperation::Execute(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                            Context *context)
{
    ATB_LOG(INFO) << "Calling Execute...";
    return opSelected_->Execute(variantPack, workspace, workspaceSize, context);
}

uint32_t IfOperation::GetInputNum() const
{
    ATB_LOG(INFO) << "Calling GetInputNum...";
    return opSelected_->GetInputNum();
}

uint32_t IfOperation::GetOutputNum() const
{
    ATB_LOG(INFO) << "Calling GetOutputNum...";
    return opSelected_->GetOutputNum();
}

void IfOperation::SetExecuteStreamId(uint32_t streamId)
{
    Status st;
    if (!opSelected_) {
        ATB_LOG(INFO) << "Operation not selected yet, setting opSelected_...";
        st = GetOperationFromCondition(&opSelected_);
        if (st != NO_ERROR) {
            return;
        }
    }
    ATB_LOG(INFO) << "Calling SetExecuteStreamId...";
    st = atb::SetExecuteStreamId(opSelected_, streamId);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "Calling SetExecuteStreamId failed!";
        return;
    }
}

Status IfOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << "Calling InferShape...";
    return opSelected_->InferShape(inTensorDescs, outTensorDescs);
}

std::shared_ptr<Runner> IfOperation::CreateRunner(Context &context) const
{
    OperationBase *opBase = dynamic_cast<OperationBase *>(opSelected_);
    if (!opBase) {
        ATB_LOG(ERROR) << "Failed to convert Operation to OperationBase";
        return nullptr;
    }
    ATB_LOG(INFO) << "Calling CreateRunner...";
    return opBase->CreateRunner(context);
}
} // namespace atb