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
#include "acltransformer/ops/post_operation.h"
#include "tests/unittest/test_util/operation_test.h"
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
using namespace AclTransformer;
using namespace AsdOps;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
TEST(TestPostOperation, InferShape)
{
    AclTransformer::PostParam param;
    AclTransformer::PostOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensors = {{AsdOps::TENSOR_DTYPE_INT64, AsdOps::TENSOR_FORMAT_ND, {1, 8}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensors, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 2);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_INT64);
    EXPECT_EQ(outTensorDescs.at(0).format, AsdOps::TENSOR_FORMAT_ND);
    AsdOps::SVector<int64_t> expectDims1 = {1, 1};
    ASSERT_EQ(expectDims1.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims1.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims1.at(1), outTensorDescs.at(0).dims.at(1));

    EXPECT_EQ(outTensorDescs.at(1).dtype, AsdOps::TENSOR_DTYPE_INT64);
    EXPECT_EQ(outTensorDescs.at(1).format, AsdOps::TENSOR_FORMAT_ND);
    AsdOps::SVector<int64_t> expectDims2 = {1, 50};
    ASSERT_EQ(expectDims2.size(), outTensorDescs.at(1).dims.size());
    EXPECT_EQ(expectDims2.at(0), outTensorDescs.at(1).dims.at(0));
    EXPECT_EQ(expectDims2.at(1), outTensorDescs.at(1).dims.at(1));
}

// AsdOps::Status PostGolden(AclTransformer::PostParam param, const GoldenContext &context)
// {
//     const Tensor &input = context.hostInTensors.at(0);
//     at::Tensor atRefinput = at::from_blob(input.data, ToIntArrayRef(input.desc.dims), at::kLong);
//     at::Tensor scores = atRefinput / param.temperature;
//     int64_t top_k = std::min(param.top_k, scores.sizes()[1]);

//     auto result = torch::sort(scores, -1, true);
//     at::Tensor valuescur = std::get<0>(result).slice(-1, 0, top_k);
//     at::Tensor indices = std::get<1>(result).slice(-1, 0, top_k);

//     at::Tensor values = torch::flip(valuescur, {1});
//     at::Tensor curprobs = torch::softmax(values, -1);
//     at::Tensor cumulative_probs = torch::cumsum(curprobs, -1);
//     at::Tensor sorted_indices_to_remove = cumulative_probs <= (1 - param.top_p);

//     if (param.min_tokens_to_keep > 1) {
//         sorted_indices_to_remove.slice(-1, -param.min_tokens_to_keep, -1) = 0;
//     }

//     at::Tensor output = torch::masked_fill(values, sorted_indices_to_remove, param.filter_value);

//     int n_dim = output.dim();
//     at::Tensor probs = torch::softmax(output, n_dim - 1);

//     int n_samples = 1;
//     bool replacement = true;
//     at::Tensor next_tokens = torch::multinomial(probs, n_samples, replacement);
//     at::Tensor atReFOutTensor = next_tokens;
//     const AsdOps::Tensor outTensor1 = context.hostOutTensors.at(0);
//     at::Tensor atOutTensor1 = at::from_blob(outTensor1.data, ToIntArrayRef(outTensor1.desc.dims), at::kLong);
//     const AsdOps::Tensor outTensor2 = context.hostOutTensors.at(1);
//     at::Tensor atOutTensor2 = at::from_blob(outTensor2.data, ToIntArrayRef(outTensor2.desc.dims), at::kLong);
//     float *atOutArray1 = (float *)atOutTensor1.storage().data_ptr().get();
//     float *atOutArray2 = (float *)atOutTensor2.storage().data_ptr().get();
//     float *atRefOutArray1 = (float *)atReFOutTensor.storage().data_ptr().get();
//     float *atRefOutArray2 = (float *)indices.storage().data_ptr().get();
//     float *outData1 = static_cast<float *>(outTensor1.data);
//     float *outData2 = static_cast<float *>(outTensor2.data);
//     for (int i = 0; i < outTensor1.Numel(); i++) {
//         float expect = atRefOutArray1[i];
//         float actual = atOutArray1[i];
//         bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
//         EXPECT_EQ(judge, true);
//         if (!judge) {
//             return Status::FailStatus(1, "unequal");
//         }
//     }
//     for (int i = 0; i < outTensor2.Numel(); i++) {
//         int expect = atRefOutArray2[i];
//         int actual = atOutArray2[i];
//         bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
//         EXPECT_EQ(judge, true);
//         if (!judge) {
//             return Status::FailStatus(1, "unequal");
//         }
//     }

//     return AsdOps::Status::OkStatus();
// }

// TEST(TestPostOperation, PostOperation)
// {
//     AclTransformer::PostParam opParam;
//     AclTransformer::PostOperation op(opParam);
//     AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {{AsdOps::TENSOR_DTYPE_INT64, AsdOps::TENSOR_FORMAT_ND, {1,
//     8}},
//                                             };
//     OperationTest opTest;
//     //opTest.LongRand(LONG_MIN_VALUE,LONG_MAX_VALUE);
//     opTest.Golden(std::bind(PostGolden, opParam, std::placeholders::_1));
//     AsdOps::Status status = opTest.Run(&op, inTensorDescs);
//     ASSERT_EQ(status.Ok(), true);
// }