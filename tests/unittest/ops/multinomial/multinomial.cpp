/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 *  * Licensed under the Apache License, Version 2.0 (the "License");
 *  * you may not use this file except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 *  * http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  */
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <half.hpp>
#include <asdops/utils/log/log.h>
#include "tests/unittest/test_util/test_common.h"

#include "acltransformer/ops/multinomial_operation.h"
#include "tests/unittest/test_util/operation_test.h"
#include <iostream>
#include <half.hpp>

#include <random>

using namespace AclTransformer;
using namespace AsdOps;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

TEST(TestMultinomial, InferShape)
{
    AclTransformer::MultinomialParam param = {3};
    AclTransformer::MultinomialOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 50}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_UINT32);
    AsdOps::SVector<int64_t> expectDims = {1, 3};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
}


AsdOps::Status MultinomialGolden(const GoldenContext &context)
{
    uint32_t numSamples = 4;
    const AsdOps::Tensor &inTensor0 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor0 =
        at::from_blob(inTensor0.data, ToIntArrayRef(inTensor0.desc.dims), at::kHalf).to(at::kFloat);

    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kInt);
    at::Tensor refOutTensor = at::multinomial(atInRefTensor0, numSamples).to(at::kInt).contiguous();
    uint32_t *result = static_cast<uint32_t *>(atOutTensor.storage().data_ptr().get());
    uint32_t *expect = static_cast<uint32_t *>(refOutTensor.storage().data_ptr().get());
    for (int i = 0; i < outTensor.Numel(); i++) {

        // EXPECT_EQ(expect[i], result[i]);
    }

    return Status::OkStatus();
}

TEST(TestMultinomial, TestMultinomial)
{

    AclTransformer::MultinomialParam param = {4};
    AclTransformer::MultinomialOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 50}}};
    OperationTest opTest;
    opTest.FloatRand(0, 1);
    opTest.Golden(&MultinomialGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}