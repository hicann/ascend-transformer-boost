/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CUSTOMIZE_BLOCK_COPY_BLOCK_COPYOPSRUNNER_H
#define CUSTOMIZE_BLOCK_COPY_BLOCK_COPYOPSRUNNER_H
#include "atb/runner/ops_runner.h"
#include "customize_op_params.h"

namespace atb {
class CustomizeBlockCopyOpsRunner : public OpsRunner {
public:
    explicit CustomizeBlockCopyOpsRunner(const customize::BlockCopyParam &param);
    ~CustomizeBlockCopyOpsRunner() override;

private:
    customize::BlockCopyParam param_;
};
} // namespace atb
#endif