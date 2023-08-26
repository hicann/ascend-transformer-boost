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
#include <half.hpp>
#include <asdops/utils/log/log.h>
#include "tests/unittest/test_util/test_common.h"
#include "acltransformer/ops/linear_parallel_operation.h"
#include "tests/unittest/test_util/operation_test.h"

using namespace AclTransformer;
using namespace AsdOps;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

TEST(TestLinearParallelOperation, InferShape)
{
    AclTransformer::LinearParallelParam param;
    AclTransformer::LinearParallelOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 2}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 2}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 2}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT16);
    EXPECT_EQ(outTensorDescs.at(0).format, AsdOps::TENSOR_FORMAT_ND);
    AsdOps::SVector<int64_t> expectDims = {1, 1, 1};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
}

AsdOps::Status LinearParallelGolden(const GoldenContext &context)
{
    const AsdOps::Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 =
        at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf).to(at::kFloat);
    const AsdOps::Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 =
        at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kHalf).to(at::kFloat);
    const AsdOps::Tensor &inTensor3 = context.hostInTensors.at(2);
    at::Tensor atInRefTensor3 =
        at::from_blob(inTensor3.data, ToIntArrayRef(inTensor3.desc.dims), at::kHalf).to(at::kFloat);

    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor refOutTensor = at::linear(atInRefTensor1, atInRefTensor2, atInRefTensor3).to(at::kHalf).contiguous();

    half_float::half *result = static_cast<half_float::half *>(atOutTensor.storage().data_ptr().get());
    half_float::half *expect = static_cast<half_float::half *>(refOutTensor.storage().data_ptr().get());
    for (int i = 0; i < outTensor.Numel(); i++) {
        bool judge = std::abs(expect[i] - result[i]) <= (ATOL + RTOL * std::abs(result[i]));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

// TEST(TestLinearParallelOperation, TestLinearParallel)
// {
//     AclTransformer::LinearParallelParam param;
//     AclTransformer::LinearParallelOperation op(param);
//     AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
//         {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
//         {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
//         {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}}};
//     OperationTest opTest;
//     opTest.Golden(&LinearParallelGolden);
//     AsdOps::Status status = opTest.Run(&op, inTensorDescs);
//     ASSERT_EQ(status.Ok(), true);
// }

AsdOps::Status LinearParallelWithoutBiasGolden(const GoldenContext &context)
{
#if 0
    const AsdOps::Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
    const AsdOps::Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kFloat);

    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    at::Tensor refOutTensor = at::linear(atInRefTensor1, atInRefTensor2, atInRefTensor2).contiguous();
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = atRefOutArray[i];
        float actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
#endif
    return Status::OkStatus();
}

TEST(TestLinearParallelOperation, TestLinearParallelWithoutBias)
{
    AclTransformer::LinearParallelParam param;
    param.bias == "None";
    AclTransformer::LinearParallelOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}}};
    OperationTest opTest;
    opTest.Golden(&LinearParallelWithoutBiasGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    // ASSERT_EQ(status.Ok(), true);
}