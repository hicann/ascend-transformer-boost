/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_RMS_NORM_ACLNN_RUNNER_H
#define ATB_RMS_NORM_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnGetWorkspaceSizeFunc = aclnnStatus (*)(
    const aclTensor *,// x
    const aclTensor *,// gamma
    double,//epsilon
    const aclTensor *,// yOut
    const aclTensor *,//rstdOut
    uint64_t *,//workspaceSize
    aclOpExecutor ** //executor
);

using AclnnExecuteFunc = aclnnStatus (*)(
    void *,//workspace
    uint64_t,//workspaceSize
    aclOpExecutor *,//executor
    aclrtStream//stream
);

namespace atb {
class RmsNormAclnnRunner : public AclnnRunner {
public:
    explicit RmsNormAclnnRunner(const infer::RmsNormParam &param);
    ~RmsNormAclnnRunner() override;

    static Status LoadMethod();
protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    aclnnStatus  CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                                 aclTensor** tensor, uint64_t dataSize);

private:
    Status CreateInputAclnnTensor();
    Status CreateGammaAclnnTensor();
    Status CreateOutputAclnnTensor();
    Status CreateRstdAclnnTensor();

private:
    infer::RmsNormParam param_;
    //两个函数指针分别对应对应aclnnop/aclnn_rms_norm.h中的两段式接口
    //GetWorkSpace接口
    static AclnnGetWorkspaceSizeFunc aclnnGetWorkspaceSizeFunc_;
    //调用接口
    static AclnnExecuteFunc aclnnExecuteFunc_;

    void* rstdDeviceAddr_ = nullptr;
    aclTensor* rstdTensor_ = nullptr;

};
} // namespace atb
#endif
