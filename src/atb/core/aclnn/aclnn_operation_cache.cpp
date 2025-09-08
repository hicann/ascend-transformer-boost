/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/core/aclnn/aclnn_operation_cache.h"
#include "atb/core/aclnn/executor_manager.h"

#include "atb/utils/log.h"
#include "atb/utils/singleton.h"

namespace atb {

void AclNNOpCache::Destroy()
{
    ATB_LOG(INFO) << "Plugin Op Cache: AclNNOpCache addr [" << (this) << "]destroy";
    if (this->aclExecutor == nullptr) {
        return;
    }

    // ExecutorManager中的引用减1
    int count = GetSingleton<ExecutorManager>().DecreaseReference(this->aclExecutor);
    if (count != 0) {
        return;
    } // 如果executor的引用不为0，则不删除executor及其对应的aclTensor

    // 如果aclExecutor存在且引用为0，则destroy
    int ret = -1;
    ATB_LOG(INFO) << "Plugin Op Cache: destroy Executor addr[" << this->aclExecutor << "]";
    if (this->executorRepeatable) {
        // 如果executor可复用，进行destroy；否则不destroy，避免对aclExecutor的重复释放
        ret = aclDestroyAclOpExecutor(this->aclExecutor);
        if (ret != 0) {
            ATB_LOG(ERROR) << "Plugin Op Cache: destroy Executor failed.";
        }
    }
    this->aclExecutor = nullptr;

    // 清空用于构造aclExecutor而创建的结构体
    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensors.size(); ++i) {
        if (this->aclnnVariantPack.aclInTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclDestroyTensor(this->aclnnVariantPack.aclInTensors[i]->tensor);
            if (ret != 0) {
                ATB_LOG(ERROR) << "Plugin Op Cache: destroy aclInTensors " << i << " failed.";
            }
        }
        ret = aclDestroyIntArray(this->aclnnVariantPack.aclInTensors[i]->intArrayHostData.intArray);
        if (ret != 0) {
            ATB_LOG(ERROR) << "Plugin Op Cache: destroy aclInTensors " << i << " intArrayHostData failed.";
        }
    }
    this->aclnnVariantPack.aclInTensors.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensors.size(); ++i) {
        if (this->aclnnVariantPack.aclOutTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclDestroyTensor(this->aclnnVariantPack.aclOutTensors[i]->tensor);
            if (ret != 0) {
                ATB_LOG(ERROR) << "Plugin Op Cache: destroy aclOutTensors " << i << " failed.";
            }
        }
    }
    this->aclnnVariantPack.aclOutTensors.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensorList.size(); ++i) {
        ret = aclDestroyTensorList(this->aclnnVariantPack.aclInTensorList[i]);
        if (ret != 0) {
            ATB_LOG(ERROR) << "Plugin Op Cache: destroy aclInTensorList " << i << " failed.";
        }
    }
    this->aclnnVariantPack.aclInTensorList.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensorList.size(); ++i) {
        ret = aclDestroyTensorList(this->aclnnVariantPack.aclOutTensorList[i]);
        if (ret != 0) {
            ATB_LOG(ERROR) << "Plugin Op Cache: destroy aclOutTensorList " << i << " failed.";
        }
    }
    this->aclnnVariantPack.aclOutTensorList.clear();
}

atb::Status AclNNOpCache::UpdateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "AclNNOpCacheUpdateAclNNVariantPack";
    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensors.size(); ++i) {
        int ret = -1;
        if (!this->aclnnVariantPack.aclInTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack.aclInTensors[i]->atbTensor = variantPack.inTensors.at(i);
        if (this->aclnnVariantPack.aclInTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetInputTensorAddr(this->aclExecutor, this->aclnnVariantPack.aclInTensors[i]->tensorIdx,
                                        this->aclnnVariantPack.aclInTensors[i]->tensor,
                                        this->aclnnVariantPack.aclInTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicInputTensorAddr(
                this->aclExecutor, this->aclnnVariantPack.aclInTensors[i]->tensorListidx,
                this->aclnnVariantPack.aclInTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclInTensorList[this->aclnnVariantPack.aclInTensors[i]->tensorListidx],
                this->aclnnVariantPack.aclInTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << "inTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensors.size(); ++i) {
        int ret = -1;
        if (!this->aclnnVariantPack.aclOutTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack.aclOutTensors[i]->atbTensor = variantPack.outTensors.at(i);
        if (this->aclnnVariantPack.aclOutTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetOutputTensorAddr(this->aclExecutor, this->aclnnVariantPack.aclOutTensors[i]->tensorIdx,
                                         this->aclnnVariantPack.aclOutTensors[i]->tensor,
                                         this->aclnnVariantPack.aclOutTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicOutputTensorAddr(
                this->aclExecutor, this->aclnnVariantPack.aclOutTensors[i]->tensorListidx,
                this->aclnnVariantPack.aclOutTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclOutTensorList[this->aclnnVariantPack.aclOutTensors[i]->tensorListidx],
                this->aclnnVariantPack.aclOutTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << "outTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    return atb::NO_ERROR;
}

} // namespace atb
