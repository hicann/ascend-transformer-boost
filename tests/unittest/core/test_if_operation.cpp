/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "test_utils/operation_test.h"
#include <vector>
#include <string>
#include <string.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include "atb/utils/tensor_util.h"
#include <atb/utils/log.h>
#include "atb/utils/probe.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cpp-stub/src/stub.h>
#include "atb/utils/singleton.h"
#include "atb/utils/config.h"

using namespace atb;
using namespace Mki;

void SimpleReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1];
}

void WrongReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum;
    newShape.dims[0] = oldShape.dims[0] + 1;
    newShape.dims[1] = oldShape.dims[1];
}

bool CondFunction(void *condition)
{
    if (condition != nullptr) {
        int *data = static_cast<int *>(condition);
        return (*data > 10);
    }
    return false;
}

TEST(TestIfhOperation, IfOpTest)
{
    atb::Operation *operationA;
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &operationA);
    EXPECT_EQ(status1, 0);

    atb::Operation *operationB;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &operationB);
    EXPECT_EQ(status2, 0);

    atb::Operation *ifOperation;
    atb::IfCondParam opCond;
    std::unique_ptr<int> data = std::make_unique<int>(15);
    opCond.handle = CondFunction;
    opCond.condition = data.get();
    opCond.opA = operationA;
    opCond.opB = operationB;
    atb::Status status3 = CreateOperation(opCond, &ifOperation);
    EXPECT_EQ(status3, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status4 = opTest.Run(ifOperation, inTensorDescs);
    EXPECT_EQ(status4, 0);

    atb::DestroyOperation(ifOperation);
}