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
#include "acltransformer/ops/rms_norm_operation.h"
#include "tests/unittest/test_util/op_test.h"

using namespace AclTransformer;
using namespace AsdOps;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

TEST(TestRmsNormOperation, InferShape)
{
    AclTransformer::RmsNormParam param;
    AclTransformer::RmsNormOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT16);
    AsdOps::SVector<int64_t> expectDims = {1, 2};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
}

AsdOps::Status RmsNormGolden(const GoldenContext &context)
{
    float rmsNormEps = 1e-12;
    const AsdOps::Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 =
        at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf).to(at::kFloat);
    const AsdOps::Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 =
        at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kHalf).to(at::kFloat);
    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);

    caffe2::TypeMeta inTensorType = atInRefTensor1.dtype();
    atInRefTensor1 = atInRefTensor1.to(torch::kFloat32);
    at::Tensor squareRslt = at::square(atInRefTensor1);
    at::Tensor variance = squareRslt.mean(-1, true);
    at::Tensor addRslt = torch::add(variance, torch::tensor(rmsNormEps));
    at::Tensor rsqrtResult = at::rsqrt(addRslt);
    at::Tensor hiddenStates = atInRefTensor1 * rsqrtResult;
    at::Tensor refOutTensor = atInRefTensor2 * hiddenStates;
    refOutTensor = refOutTensor.to(inTensorType).contiguous();

    half_float::half *result = static_cast<half_float::half *>(atOutTensor.storage().data_ptr().get());
    half_float::half *expect = static_cast<half_float::half *>(refOutTensor.storage().data_ptr().get());
    for (int i = 0; i < outTensor.Numel(); i++) {
        bool judge = std::abs(expect[i] - result[i]) <= (ATOL + RTOL * std::abs(result[i]));
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

TEST(TestRmsNormOperation, TestRmsNorm)
{
    AclTransformer::RmsNormParam param;
    AclTransformer::RmsNormOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}}};
    OpTest opTest;
    opTest.Golden(&RmsNormGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
}