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
#include "acltransformer/ops/self_attention_operation.h"
#include "tests/unittest/test_util/op_test.h"

using namespace AclTransformer;
using namespace AsdOps;

TEST(TestSelfAttentionOperation, InferShape) {
    AclTransformer::SelfAttentionParam param;
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    // openbert
    param.model = "openbert";
    AclTransformer::SelfAttentionOperation opOpenbert(param);
    opOpenbert.InferShape(inTensorDescs, outTensorDescs);
    AsdOps::SVector<int64_t> expectDims = {1, 1, 1, 1};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(0).dims.at(3));

    // chatglm6b / glm130b / chatglm2_6b
    param.model = "chatglm6b";
    AclTransformer::SelfAttentionOperation opChatglm6b(param);
    opChatglm6b.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 4);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);

    param.model = "glm130b";
    AclTransformer::SelfAttentionOperation opGlm130b(param);
    opGlm130b.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 4);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);

    param.model = "chatglm2_6b";
    AclTransformer::SelfAttentionOperation opChatglm2_6b(param);
    opChatglm2_6b.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 4);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);

    // llama7b
    param.model = "llama7b";
    AclTransformer::SelfAttentionOperation opLlama7b(param);
    opLlama7b.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
}

// AsdOps::Status AddGolden(const GoldenContext &context) {
//     return Status::OkStatus();
// }

// TEST(TestSelfAttentionOperation, TestSelfAttention) {
//     AclTransformer::SelfAttentionParam param;
//     AclTransformer::SelfAttentionOperation op(param);
//     AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2}},
//                                                          {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2}}};

//     OpTest opTest(3);
//     opTest.Golden(&AddGolden);
//     AsdOps::Status status = opTest.Run(&op, inTensorDescs);
//     ASSERT_EQ(status.Ok(), true);
// }