/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_ops_runner.h"
#include "atb/utils/log.h"

namespace atb {
static const uint32_t NUMONE = 1;
static const uint32_t NUMTWO = 2;
static const uint32_t NUMTHREE = 3;
static const uint32_t INDEX_ZERO = 0;
static const uint32_t INDEX_ONE = 1;
static const uint32_t INDEX_TWO = 2;

FusionOpsRunner::FusionOpsRunner(const infer::FusionParam &param)
    : OpsRunner("FusionOpsRunner", RUNNER_TYPE_FUSION), param_(param)
{
    ATB_LOG(INFO) << "FusionOpsRunner::FusionOpsRunner called";

    kernelGraph_.nodes.resize(NUMONE);
    auto &fusionNode = kernelGraph_.nodes.at(INDEX_ZERO);
    if (!SetIntensor(fusionNode)) {
        return;
    }
    SetOuttensor(fusionNode);
    ATB_LOG(INFO) << "FusionOpsRunner::FusionOpsRunner end";
}

FusionOpsRunner::~FusionOpsRunner() {}

bool FusionOpsRunner::SetIntensor(KernelGraphNode &fusionNode)
{
    uint32_t inTensorNum = GetIntensorSize();
    if (inTensorNum == 0) {
        return false;
    }
    kernelGraph_.inTensors.resize(inTensorNum);
    if (inTensorNum == NUMONE) {
        Mki::Tensor &aTensor = kernelGraph_.inTensors.at(INDEX_ZERO);
        fusionNode.inTensors = {&aTensor};
    } else if (inTensorNum == NUMTWO) {
        Mki::Tensor &aTensor = kernelGraph_.inTensors.at(INDEX_ZERO);
        Mki::Tensor &bTensor = kernelGraph_.inTensors.at(INDEX_ONE);
        fusionNode.inTensors = {&aTensor, &bTensor};
    } else if (inTensorNum == NUMTHREE) {
        Mki::Tensor &aTensor = kernelGraph_.inTensors.at(INDEX_ZERO);
        Mki::Tensor &bTensor = kernelGraph_.inTensors.at(INDEX_ONE);
        Mki::Tensor &cTensor = kernelGraph_.inTensors.at(INDEX_TWO);
        fusionNode.inTensors = {&aTensor, &bTensor, &cTensor};
    } else {
        ATB_LOG(WARN) << "FusionOpsRunner::FusionOpsRunner inTensorNum: " << inTensorNum;
    }
    return true;
}

bool FusionOpsRunner::SetOuttensor(KernelGraphNode &fusionNode)
{
    AtbOps::OpParam::Fusion::FusionType opFusionType = GetOpFusionType();
    AtbOps::OpParam::Fusion fusionParam = {opFusionType};
    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &operationOutTensor0 = kernelGraph_.outTensors.at(INDEX_ZERO);
    fusionNode.outTensors = {&operationOutTensor0};
    fusionParam.outTensorType = GetOutTensorType(param_.outTensorType);
    fusionNode.opDesc = {0, "FusionOperation", fusionParam};
    if (fusionParam.fusionType == AtbOps::OpParam::Fusion::MATMUL_ADD) {
        fusionNode.opDesc = {1, "FusionOperation", fusionParam};
    } else if (fusionParam.fusionType == AtbOps::OpParam::Fusion::MATMUL_GELU) {
        fusionNode.opDesc = {2, "FusionOperation", fusionParam};
    } else if (fusionParam.fusionType == AtbOps::OpParam::Fusion::MATMUL_SIGMOID) {
        fusionNode.opDesc = {3, "FusionOperation", fusionParam};
    } else if (fusionParam.fusionType == AtbOps::OpParam::Fusion::MATMUL_SWIGLU) {
        fusionNode.opDesc = {4, "FusionOperation", fusionParam};
    }
}

uint32_t FusionOpsRunner::GetIntensorSize() const
{
    static std::map<infer::FusionParam::FusionType, uint32_t> inTensorNumTable = {
        {infer::FusionParam::FusionType::MATMUL_ADD, NUMTHREE},
        {infer::FusionParam::FusionType::MATMUL_GELU, NUMTWO},
        {infer::FusionParam::FusionType::MATMUL_SIGMOID, NUMTWO},
        {infer::FusionParam::FusionType::MATMUL_SWIGLU, NUMTWO},
    };
    std::map<infer::FusionParam::FusionType, uint32_t>::const_iterator it = inTensorNumTable.find(param_.fusionType);
    return it == inTensorNumTable.end() ? 0 : it->second;
}

AtbOps::OpParam::Fusion::FusionType FusionOpsRunner::GetOpFusionType() const
{
    static std::map<infer::FusionParam::FusionType, AtbOps::OpParam::Fusion::FusionType> typeTable = {
        {infer::FusionParam::FusionType::MATMUL_ADD, AtbOps::OpParam::Fusion::MATMUL_ADD},
        {infer::FusionParam::FusionType::MATMUL_GELU, AtbOps::OpParam::Fusion::MATMUL_GELU},
        {infer::FusionParam::FusionType::MATMUL_SIGMOID, AtbOps::OpParam::Fusion::MATMUL_SIGMOID},
        {infer::FusionParam::FusionType::MATMUL_SWIGLU, AtbOps::OpParam::Fusion::MATMUL_SWIGLU},
    };
    std::map<infer::FusionParam::FusionType, AtbOps::OpParam::Fusion::FusionType>::const_iterator it =
        typeTable.find(param_.fusionType);
    return it == typeTable.end() ? AtbOps::OpParam::Fusion::MATMUL_ADD : it->second; // NON_FUSION
}

Mki::TensorDType FusionOpsRunner::GetOutTensorType(const aclDataType outType) const
{
    static std::map<aclDataType, Mki::TensorDType> typeTable = {
        {aclDataType::ACL_INT8, Mki::TensorDType::TENSOR_DTYPE_INT8},
        {aclDataType::ACL_FLOAT, Mki::TensorDType::TENSOR_DTYPE_FLOAT},
        {aclDataType::ACL_FLOAT16, Mki::TensorDType::TENSOR_DTYPE_FLOAT16},
        {aclDataType::ACL_INT32, Mki::TensorDType::TENSOR_DTYPE_INT32},
        {aclDataType::ACL_INT64, Mki::TensorDType::TENSOR_DTYPE_INT64},
        {aclDataType::ACL_BF16, Mki::TensorDType::TENSOR_DTYPE_BF16},
    };
    std::map<aclDataType, Mki::TensorDType>::const_iterator it = typeTable.find(outType);
    return it == typeTable.end() ? Mki::TensorDType::TENSOR_DTYPE_UNDEFINED : it->second;
}
} // namespace atb