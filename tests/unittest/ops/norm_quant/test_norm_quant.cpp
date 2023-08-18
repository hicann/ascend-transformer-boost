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
#include "acltransformer/ops/norm_quant_operation.h"
#include "tests/unittest/test_util/op_test.h"

using namespace AclTransformer;
using namespace AsdOps;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
constexpr int QUANTMAX = 127;
constexpr int QUANTMIN = -128;
constexpr float SCALE = 1;
constexpr int OFFSET = 0;
constexpr float ALPHA = 1;
TEST(TestNormQuantOperation, InferShape)
{
    AclTransformer::NormQuantParam param;
    AclTransformer::NormQuantOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 2);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_INT8);
    EXPECT_EQ(outTensorDescs.at(1).dtype, AsdOps::TENSOR_DTYPE_FLOAT16);
    AsdOps::SVector<int64_t> expectDims[2] = {{1, 2}, {1, 2}};
    ASSERT_EQ(expectDims[0].size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims[0].at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims[0].at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims[1].at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims[1].at(1), outTensorDescs.at(0).dims.at(1));
}

AsdOps::Status NormQuantGolden(const GoldenContext &context)
{
    int countFalse = 0;
    double layerNormEps = 1e-12;
    const AsdOps::Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 =
        at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf).to(at::kFloat);
    const AsdOps::Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 =
        at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kHalf).to(at::kFloat);
    const AsdOps::Tensor &inTensor3 = context.hostInTensors.at(2);
    at::Tensor atInRefTensor3 =
        at::from_blob(inTensor3.data, ToIntArrayRef(inTensor3.desc.dims), at::kHalf).to(at::kFloat);

    const AsdOps::Tensor outTensor0 = context.hostOutTensors.at(0);
    const AsdOps::Tensor outTensor1 = context.hostOutTensors.at(1);
    at::Tensor atOutTensor0 = at::from_blob(outTensor0.data, ToIntArrayRef(outTensor0.desc.dims), at::kChar);
    at::Tensor atOutTensor1 = at::from_blob(outTensor1.data, ToIntArrayRef(outTensor1.desc.dims), at::kHalf);
    at::Tensor refOutTensor =
        at::layer_norm(atInRefTensor1, atInRefTensor2.sizes(), atInRefTensor2, atInRefTensor3, layerNormEps)
            .to(at::kHalf);

    auto groundtruth1 = at::mul(refOutTensor, ALPHA).to(at::kHalf);
    auto res1 = at::mul(refOutTensor, SCALE);
    auto groundtruth0 = at::add(res1, OFFSET);
    
    int8_t *result0 = static_cast<int8_t *>(atOutTensor0.storage().data_ptr().get());
    half_float::half *expect0 = static_cast<half_float::half *>(groundtruth0.storage().data_ptr().get());
    for (int i = 0; i < outTensor0.Numel(); i++) {
        int res1 = result0[i];
        int res2 = std::round(expect0[i]);
        res2 = std::min(std::max(res2, QUANTMIN), QUANTMAX);
        if (res1 != res2) {
            if (abs(abs(res1) - abs(res2)) == 1) {
                countFalse = countFalse + 1;
            }
        }
    }
    ASD_LOG(INFO) << "countFalse = " << countFalse;
    half_float::half *result = static_cast<half_float::half *>(atOutTensor1.storage().data_ptr().get());
    half_float::half *expect = static_cast<half_float::half *>(groundtruth1.storage().data_ptr().get());
    for (int i = 0; i < outTensor0.Numel(); i++) {
        bool judge = std::abs(expect[i] - result[i]) <= (ATOL + RTOL * std::abs(result[i]));
        // EXPECT_EQ(judge, true);
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

TEST(TestNormQuantOperation, TestNormQuant)
{
    AclTransformer::NormQuantParam param;
    AclTransformer::NormQuantOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 32}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 32}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 32}}};
    OpTest opTest;
    opTest.Golden(&NormQuantGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    // ASSERT_EQ(status.Ok(), true);
}