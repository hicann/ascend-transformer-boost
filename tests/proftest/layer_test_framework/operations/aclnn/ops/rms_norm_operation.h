/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef ATB_SPEED_PLUGIN_ACLNN_RMSNORM_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_RMSNORM_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
class RmsNormOperation : public AclNNOperation {
public:
    explicit RmsNormOperation(const std::string &name, float epsilon);
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    float epsilon = 1e-5;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
};
} // namespace common
} // namespace atb_speed
#endif