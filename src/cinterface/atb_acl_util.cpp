/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/atb_acl.h"
#include "atb/utils/log.h"

#ifdef __cplusplus
extern "C" {
#endif
int64_t GetTensorSize(const aclTensor *input)
{
    const op::Shape shape = input->GetViewShape();
    const size_t dims = shape.GetDimNum();
    int64_t size = 1;
    for (size_t i = 0; i < dims; ++i) {
        size *= shape.GetDim(i);
    }
    return size;
}

atb::Status aclTensorToAtbTensor(const aclTensor *aclTensorSrc, atb::Tensor *atbTensorDst)
{
    if (aclTensorSrc == nullptr) {
        atbTensorDst->hostData = nullptr;
        atbTensorDst->deviceData = nullptr;
        return atb::NO_ERROR;
    }
    int64_t *dims = nullptr;
    uint64_t dimCount;
    aclDataType dataType;
    aclFormat format;
    auto status = aclGetViewShape(aclTensorSrc, &dims, &dimCount);
    ATB_CHECK(status == atb::NO_ERROR, "aclGetViewShape failed!", return status);
    status = aclGetDataType(aclTensorSrc, &dataType);
    ATB_CHECK(status == atb::NO_ERROR, "aclGetDataType failed!", return status);
    status = aclGetFormat(aclTensorSrc, &format);
    ATB_CHECK(status == atb::NO_ERROR, "aclGetFormat failed!", return status);
    atb::TensorDesc desc;
    desc.shape.dimNum = dimCount;
    for (size_t i = 0; i < dimCount; i++) {
        desc.shape.dims[i] = (static_cast<int64_t *>(dims))[i];
    }
    desc.format = format;
    desc.dtype = dataType;
    atbTensorDst->desc = desc;
    atbTensorDst->deviceData = aclTensorSrc->GetData();
    atbTensorDst->hostData = nullptr;
    atbTensorDst->dataSize = GetTensorSize(aclTensorSrc) * aclDataTypeSize(dataType);
    return atb::NO_ERROR;
}

atb::Status aclTensorToAtbTensorHost(const aclTensor *aclTensorSrc, atb::Tensor *atbTensorDst)
{
    if (aclTensorSrc == nullptr) {
        atbTensorDst->hostData = nullptr;
        atbTensorDst->deviceData = nullptr;
        return atb::NO_ERROR;
    }
    int64_t *dims = nullptr;
    uint64_t dimCount;
    aclDataType dataType;
    aclFormat format;
    auto status = aclGetViewShape(aclTensorSrc, &dims, &dimCount);
    ATB_CHECK(status == atb::NO_ERROR, "aclGetViewShape failed!", return status);
    status = aclGetDataType(aclTensorSrc, &dataType);
    ATB_CHECK(status == atb::NO_ERROR, "aclGetDataType failed!", return status);
    status = aclGetFormat(aclTensorSrc, &format);
    ATB_CHECK(status == atb::NO_ERROR, "aclGetFormat failed!", return status);
    atb::TensorDesc desc;
    desc.shape.dimNum = dimCount;
    for (size_t i = 0; i < dimCount; i++) {
        desc.shape.dims[i] = (static_cast<int64_t *>(dims))[i];
    }
    desc.format = format;
    desc.dtype = dataType;
    atbTensorDst->desc = desc;
    atbTensorDst->deviceData = nullptr;
    atbTensorDst->hostData = aclTensorSrc->GetData();
    atbTensorDst->dataSize = GetTensorSize(aclTensorSrc) * aclDataTypeSize(dataType);
    return atb::NO_ERROR;
}

#ifdef __cplusplus
}
#endif
