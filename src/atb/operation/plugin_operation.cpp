/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation/plugin_operation.h"
#include "atb/runner/plugin_runner.h"

namespace atb {
PluginOperation::PluginOperation(Operation *operation) : OperationBase("PluginOperation")
{
    operation_.reset(operation);
}

PluginOperation::~PluginOperation() {}

std::string PluginOperation::GetName() const
{
    if (operation_) {
        return operation_->GetName();
    } else {
        return "PluginOperation";
    }
}

uint32_t PluginOperation::GetInputNum() const
{
    if (operation_) {
        return operation_->GetInputNum();
    }
    return 0;
}

uint32_t PluginOperation::GetOutputNum() const
{
    if (operation_) {
        return operation_->GetOutputNum();
    }
    return 0;
}

Status PluginOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    if (operation_) {
        return operation_->InferShape(inTensorDescs, outTensorDescs);
    }
    return ERROR_INVALID_PARAM;
}

std::shared_ptr<Runner> PluginOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<PluginRunner>(operation_.get());
}
} // namespace atb