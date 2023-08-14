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
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    AsdOps::SVector<int64_t> expectDims;
    
    // openbert(default)
    AclTransformer::SelfAttentionOperation op0(param);
    op0.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    expectDims = {1, 2, 3, 4};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(0).dims.at(3));

    // chatglm6b / glm130b / chatglm2_6b
    param.model = "chatglm6b";
    AclTransformer::SelfAttentionOperation op1(param);
    op1.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    expectDims = {1, 2, 12};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));

    param.model = "glm130b";
    AclTransformer::SelfAttentionOperation op2(param);
    op2.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    expectDims = {1, 2, 12};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));

    param.model = "chatglm2_6b";
    AclTransformer::SelfAttentionOperation op3(param);
    op3.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    expectDims = {1, 2, 12};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));

    // llama7b
    param.model = "llama7b";
    AclTransformer::SelfAttentionOperation op4(param);
    op4.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<AsdOps::SVector<int64_t>> expectDimsLlama7b = {{1, 2, 12}, {2, 3, 4, 5}, {3, 4, 5, 6}};
    ASSERT_EQ(expectDimsLlama7b.at(0).size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDimsLlama7b.at(0).at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDimsLlama7b.at(0).at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDimsLlama7b.at(0).at(2), outTensorDescs.at(0).dims.at(2));
    ASSERT_EQ(expectDimsLlama7b.at(1).size(), outTensorDescs.at(1).dims.size());
    EXPECT_EQ(expectDimsLlama7b.at(1).at(0), outTensorDescs.at(1).dims.at(0));
    EXPECT_EQ(expectDimsLlama7b.at(1).at(1), outTensorDescs.at(1).dims.at(1));
    EXPECT_EQ(expectDimsLlama7b.at(1).at(2), outTensorDescs.at(1).dims.at(2));
    EXPECT_EQ(expectDimsLlama7b.at(1).at(3), outTensorDescs.at(1).dims.at(3));
    ASSERT_EQ(expectDimsLlama7b.at(2).size(), outTensorDescs.at(2).dims.size());
    EXPECT_EQ(expectDimsLlama7b.at(2).at(0), outTensorDescs.at(2).dims.at(0));
    EXPECT_EQ(expectDimsLlama7b.at(2).at(1), outTensorDescs.at(2).dims.at(1));
    EXPECT_EQ(expectDimsLlama7b.at(2).at(2), outTensorDescs.at(2).dims.at(2));
    EXPECT_EQ(expectDimsLlama7b.at(2).at(3), outTensorDescs.at(2).dims.at(3));
}

// AsdOps::Status AddGolden(const GoldenContext &context) {
//     const AsdOps::Tensor &inTensor1 = context.hostInTensors.at(0);
//     at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
//     const AsdOps::Tensor &inTensor2 = context.hostInTensors.at(1);
//     at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kFloat);

//     torch::Tensor mixedQuery = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[0]);
//     mixedQuery = mixedQuery.view({mixedQuery.sizes()[0], mixedQuery.sizes()[1] * this->param_.headNum,
//                                   mixedQuery.sizes()[2] / this->param_.headNum});
//     mixedQuery = torch::transpose(mixedQuery, 0, 1);
//     torch::Tensor mixedKey = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[1]);
//     torch::Tensor mixedValue = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[2]);
//     mixedValue = mixedValue.view({mixedValue.sizes()[0], mixedValue.sizes()[1] * this->param_.headNum,
//                                   mixedValue.sizes()[2] / this->param_.headNum});
//     mixedValue = torch::transpose(mixedValue, 0, 1);
//     mixedKey = mixedKey.view(
//         {mixedKey.sizes()[0], mixedKey.sizes()[1] * this->param_.headNum, mixedKey.sizes()[2] / this->param_.headNum});
//     mixedKey = mixedKey.permute({1, 2, 0});

//     torch::Tensor attention_mask = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[3]);

//     double scal = 1 / sqrt(this->param_.dk);
//     torch::Tensor attentionScores = torch::bmm(mixedQuery, mixedKey).contiguous();
//     attentionScores = torch::mul(attentionScores, scal);
//     attentionScores = attentionScores.view({attentionScores.sizes()[0] / this->param_.headNum, this->param_.headNum,
//                                             attentionScores.sizes()[1], attentionScores.sizes()[2]});
//     attentionScores = torch::add(attentionScores, attention_mask);
//     attentionScores = attentionScores.view({attentionScores.sizes()[0] * attentionScores.sizes()[1],
//                                             attentionScores.sizes()[2], attentionScores.sizes()[3]});

//     torch::Tensor attention_probs = torch::softmax(attentionScores, -1);
//     torch::Tensor contextLayer = torch::bmm(attention_probs, mixedValue);
//     contextLayer = torch::transpose(contextLayer, 0, 1).contiguous();
//     torch::Tensor atOutTensor = contextLayer
//                                     .view({contextLayer.sizes()[0], contextLayer.sizes()[1] / this->param_.headNum,
//                                            contextLayer.sizes()[2] * this->param_.headNum})
//                                     .contiguous();

//     TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, runnerVariantPack.outTensors[0]);
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