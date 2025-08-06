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
#include "matmul_allreduce_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_all_reduce.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

MatmulAllreduceOperation::MatmulAllreduceOperation(const std::string &name, HcclComm hcommInfo)
    : AclNNOperation(name), hcommInfo_(hcommInfo)
{
    HcclGetCommName(hcommInfo_, this->hcommName);
}

MatmulAllreduceOperation::~MatmulAllreduceOperation() {}

atb::Status MatmulAllreduceOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

    if (inTensorDescs.at(0).shape.dimNum == DIM3) {
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM2] = inTensorDescs.at(DIM1).shape.dims[DIM0];
    } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
        outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[DIM1];
    } else {
        ATB_SPEED_LOG_ERROR(opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum);
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t MatmulAllreduceOperation::GetInputNum() const { return NUM2; }

uint32_t MatmulAllreduceOperation::GetOutputNum() const { return NUM1; }

int MatmulAllreduceOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
    int ret = 0;
    ret = CreateAclNNInTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNInTensorVariantPack fail");
        return ret;
    }
    ret = CreateAclNNOutTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNOutTensorVariantPack fail");
        return ret;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return atb::NO_ERROR;
}

int MatmulAllreduceOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = -1;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.inTensors.at(i);

        if (false) {
            aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
            std::vector<int64_t> shapeT(4); // 4: dimNu
            shapeT[DIM0] = squeezedAtbTensor.desc.shape.dims[1] / 16; // 16: NZ_FORMAT
            shapeT[DIM1] = squeezedAtbTensor.desc.shape.dims[0] / 16; // 16: NZ_FORMAT
            shapeT[DIM2] = 16; // 16: NZ_FORMAT
            shapeT[DIM3] = 16; // 16: NZ_FORMAT
            aclnnTensor->tensor = aclCreateTensor(shapeT.data(),
                4, // 4: dimNum
                squeezedAtbTensor.desc.dtype,
                aclnnTensor->strides.data(),
                0,
                squeezedAtbTensor.desc.format,
                shapeT.data(),
                4, // 4: dimNum
                squeezedAtbTensor.deviceData);
        } else {
            aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
            aclnnTensor->tensor = aclCreateTensor(squeezedAtbTensor.desc.shape.dims,
                squeezedAtbTensor.desc.shape.dimNum,
                squeezedAtbTensor.desc.dtype,
                aclnnTensor->strides.data(),
                0,
                squeezedAtbTensor.desc.format,
                squeezedAtbTensor.desc.shape.dims,
                squeezedAtbTensor.desc.shape.dimNum,
                squeezedAtbTensor.deviceData);
        }
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int MatmulAllreduceOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = -1;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor atbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.desc.dtype,
            aclnnTensor->strides.data(), 0, atbTensor.desc.format, atbTensor.desc.shape.dims,
            atbTensor.desc.shape.dimNum, atbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int MatmulAllreduceOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnMatmulAllReduceGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnMatmulAllReduceGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
        nullptr,
        this->hcommName,
        "sum",
        0,
        1,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnMatmulAllReduceGetWorkspaceSize end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MatmulAllreduceOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnMatmulAllReduce start");
    int ret = aclnnMatmulAllReduce(workspace, this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnMatmulAllReduce end, ret:" << ret);
    return ret;
}

} // namespace common
} // namespace atb_speed
