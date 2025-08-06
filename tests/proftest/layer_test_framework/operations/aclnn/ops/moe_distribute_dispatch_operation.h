/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_DESTRIBUTE_DISPATCH_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_DESTRIBUTE_DISPATCH_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct MoeDistributeDispatchParam {
    int32_t epRankId = 0;
    int32_t epRankSize = 1;
    int32_t tpRankId = 0;
    int32_t tpRankSize = 1;
    int32_t expertSharedType = 0;
    int32_t maxDecodeDpTokenSize = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 1;
    int64_t localMoeExpertNum = 1;
    int64_t topk = 8;
    int64_t quantMode = 2;
    int64_t globalBS = 0; // tiling里处理成BS*world_size
    int64_t expertTokenNumsType = 0;
    bool isQuant = false;
    bool isSharedExpert = false;
    bool quantSmooth = false;
    std::string tpCommName;
    std::string epCommName;
    std::string rankTableFile = "";
};

class MoeDistributeDispatchOperation : public AclNNOperation {
public:
    explicit MoeDistributeDispatchOperation(const std::string &name, MoeDistributeDispatchParam param);
    ~MoeDistributeDispatchOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    int32_t GetGlobalBS(const atb::TensorDesc &inTensorDesc) const;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    MoeDistributeDispatchParam param_;
};

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_DESTRIBUTE_DISPATCH__OPERATION_H