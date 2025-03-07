/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_all_operation.h"
#include "atb/utils/config.h"
#include "all_to_all_hccl_runner.h"
#include "all_to_all_lccl_runner.h"
#include "atb/utils.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/core/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 1;

template <> Status CreateOperation(const infer::AllToAllParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.backend != "hccl" && opParam.backend != "lccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must be hccl or lccl";
        return ERROR_INVALID_PARAM;
    }
    const char *socName = aclrtGetSocName();
    if (!socName) {
        ATB_LOG(ERROR) << "aclrtGetSocName failed!";
        return false;
    }
    if (opParam.backend == "hccl" && !GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "AllToAll hccl only supports Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lccl" && opParam.rankSize / 2 != 0) { // 2 : Even ranksize
        ATB_LOG(ERROR) << "AllToAll lccl only supports even ranksize";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lccl" && std::string(socName).find("Ascend910_93") == std::string::npos) {
        ATB_LOG(ERROR) << "AllToAll lccl only supports Atlas 800T A3 or Atlas 900 A3 Superpod";
        return ERROR_INVALID_PARAM;
    }
    if (OperationUtil::DistributedInitCheck<infer::AllToAllParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "AllToAllOperation DistributedInitCheck failed";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) AllToAllOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new AllToAllOperation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

AllToAllOperation::AllToAllOperation(const infer::AllToAllParam &param)
    : OperationBase("AllToAllOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("AllToAllOperation");
}

AllToAllOperation::~AllToAllOperation() {}

uint32_t AllToAllOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t AllToAllOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status AllToAllOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                         SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return 0;
}

Status AllToAllOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (!TensorUtil::TensorDescEqual(inTensors.at(0).desc, outTensors.at(0).desc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor desc and outtensor desc should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> AllToAllOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<AllToAllHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<AllToAllHcclRunner>(param_, param_.hcclComm);
        }
    } else if (param_.backend == "lccl") {
        return std::make_shared<AllToAllLcclRunner>(param_);
    }
    return std::shared_ptr<Runner>();
}

nlohmann::json AllToAllOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb