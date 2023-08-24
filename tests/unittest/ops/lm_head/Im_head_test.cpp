/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <asdops/utils/log/log.h>
#include "tests/unittest/test_util/test_common.h"
#include "acltransformer/ops/lm_head_operation.h"
#include "tests/unittest/test_util/operation_test.h"
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
using namespace AclTransformer;
using namespace AsdOps;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
TEST(TestLmHeadOperation, InferShape)
{
    AclTransformer::LmHeadParam param;
    AclTransformer::LmHeadOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensors = {{AsdOps::TENSOR_DTYPE_INT8, AsdOps::TENSOR_FORMAT_ND, {1, 8}},
                                                 {AsdOps::TENSOR_DTYPE_INT8, AsdOps::TENSOR_FORMAT_ND, {1, 8}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensors, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_INT8);
    EXPECT_EQ(outTensorDescs.at(0).format, AsdOps::TENSOR_FORMAT_ND);
    AsdOps::SVector<int64_t> expectDims1 = {8, 1, 1};
    ASSERT_EQ(expectDims1.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims1.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims1.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims1.at(1), outTensorDescs.at(0).dims.at(2));
}
AsdOps::Status LmHeadGolden(const GoldenContext &context)
{
    const Tensor &input = context.hostInTensors.at(0);
    at::Tensor atInRefinput = at::from_blob(input.data, ToIntArrayRef(input.desc.dims), at::kFloat);

    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    // float *atRefOutArray = (float *)refOutTensor.storage().data_ptr().get();
    float *outData = static_cast<float *>(outTensor.data);
    // for (int i = 0; i < outTensor.Numel(); i++) {
    //     float expect = atRefOutArray[i];
    //     float actual = atOutArray[i];
    //     bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
    //     EXPECT_EQ(judge, true);
    //     if (!judge) {
    //         return Status::FailStatus(1, "unequal");
    //     }
    // }
    return Status::OkStatus();
}
TEST(TestLmHeadOperation, LmHeadOperation)
{
    AclTransformer::LmHeadParam opParam;
    AclTransformer::LmHeadOperation op(opParam);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3}}};
    OperationTest opTest;
    opTest.Golden(&LmHeadGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}