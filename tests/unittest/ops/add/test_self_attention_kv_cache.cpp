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
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "tests/unittest/test_util/op_test.h"

using namespace AclTransformer;
using namespace AsdOps;

TEST(TestSelfAttentionKvCacheOperation, InferShape)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {5, 6, 7, 8}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {6, 7, 8, 9}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    AsdOps::SVector<AsdOps::SVector<int64_t>> expectDims;

    // default(chatglm6b, not chatglm2_6b)
    AclTransformer::SelfAttentionKvCacheOperation op0(param);
    op0.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    expectDims = {{1, 2, 12}, {6, 6, 7, 8}, {7, 7, 8, 9}};
    ASSERT_EQ(expectDims.at(0).size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0).at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(0).at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(0).at(2), outTensorDescs.at(0).dims.at(2));
    ASSERT_EQ(expectDims.at(1).size(), outTensorDescs.at(1).dims.size());
    EXPECT_EQ(expectDims.at(1).at(0), outTensorDescs.at(1).dims.at(0));
    EXPECT_EQ(expectDims.at(1).at(1), outTensorDescs.at(1).dims.at(1));
    EXPECT_EQ(expectDims.at(1).at(2), outTensorDescs.at(1).dims.at(2));
    EXPECT_EQ(expectDims.at(1).at(3), outTensorDescs.at(1).dims.at(3));
    ASSERT_EQ(expectDims.at(2).size(), outTensorDescs.at(2).dims.size());
    EXPECT_EQ(expectDims.at(2).at(0), outTensorDescs.at(2).dims.at(0));
    EXPECT_EQ(expectDims.at(2).at(1), outTensorDescs.at(2).dims.at(1));
    EXPECT_EQ(expectDims.at(2).at(2), outTensorDescs.at(2).dims.at(2));
    EXPECT_EQ(expectDims.at(2).at(3), outTensorDescs.at(2).dims.at(3));

    // chatglm2_6b
    param.model = "chatglm2_6b";
    AclTransformer::SelfAttentionKvCacheOperation op1(param);
    op1.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    expectDims = {{1, 2, 12}, {6, 6, 7, 8}, {7, 7, 8, 9}};
    ASSERT_EQ(expectDims.at(0).size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0).at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(0).at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(0).at(2), outTensorDescs.at(0).dims.at(2));
    ASSERT_EQ(expectDims.at(1).size(), outTensorDescs.at(1).dims.size());
    EXPECT_EQ(expectDims.at(1).at(0), outTensorDescs.at(1).dims.at(0));
    EXPECT_EQ(expectDims.at(1).at(1), outTensorDescs.at(1).dims.at(1));
    EXPECT_EQ(expectDims.at(1).at(2), outTensorDescs.at(1).dims.at(2));
    EXPECT_EQ(expectDims.at(1).at(3), outTensorDescs.at(1).dims.at(3));
    ASSERT_EQ(expectDims.at(2).size(), outTensorDescs.at(2).dims.size());
    EXPECT_EQ(expectDims.at(2).at(0), outTensorDescs.at(2).dims.at(0));
    EXPECT_EQ(expectDims.at(2).at(1), outTensorDescs.at(2).dims.at(1));
    EXPECT_EQ(expectDims.at(2).at(2), outTensorDescs.at(2).dims.at(2));
    EXPECT_EQ(expectDims.at(2).at(3), outTensorDescs.at(2).dims.at(3));
}