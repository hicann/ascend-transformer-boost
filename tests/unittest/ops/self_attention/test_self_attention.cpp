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
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

/// @brief openbert(default) infershape test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, InferShapeOpenbert)
{
    AclTransformer::SelfAttentionParam param;
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 2}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<int64_t> expectDims = {1, 2, 3};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
}

/// @brief chatglm6b infershape test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, InferShapeChatglm6b)
{
    AclTransformer::SelfAttentionParam param;
    param.model = "chatglm6b";
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<int64_t> expectDims = {1, 2, 12};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
}

/// @brief glm130b infershape test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, InferShapeGlm130b)
{
    AclTransformer::SelfAttentionParam param;
    param.model = "glm130b";
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<int64_t> expectDims = {1, 2, 12};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
}

/// @brief chatglm2_6b infershape test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, InferShapeChatglm2_6b)
{
    AclTransformer::SelfAttentionParam param;
    param.model = "chatglm2_6b";
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<int64_t> expectDims = {1, 2, 12};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
}

/// @brief llama7b infershape test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, InferShapeLlama7b)
{
    AclTransformer::SelfAttentionParam param;
    param.model = "llama7b";
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    AsdOps::SVector<AsdOps::SVector<int64_t>> expectDims = {{1, 2, 12}, {2, 3, 4, 5}, {3, 4, 5, 6}};
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

/// @brief openbert golden
/// @param context
/// @return
AsdOps::Status SelfAttentionOpenbertGolden(const GoldenContext &context)
{
    // define param
    int64_t paramHeadNum = 1;
    int64_t paramDk = 1;
    // get constructed input/output tensors
    const AsdOps::Tensor inTensor1 = context.hostInTensors.at(0);   // {1, 2, 3}
    const AsdOps::Tensor inTensor2 = context.hostInTensors.at(1);   // {2, 2, 3}
    const AsdOps::Tensor inTensor3 = context.hostInTensors.at(2);   // {2, 2, 3}
    const AsdOps::Tensor inTensor4 = context.hostInTensors.at(3);   // {1, 1, 2}
    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);  // {1, 2, 3}
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // construct ref input tensors
    at::Tensor atInRefMixedQuery = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
    at::Tensor atInRefMixedKey = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kFloat);
    at::Tensor atInRefMixedValue = at::from_blob(inTensor3.data, ToIntArrayRef(inTensor3.desc.dims), at::kFloat);
    at::Tensor atInRefAttentionMask = at::from_blob(inTensor4.data, ToIntArrayRef(inTensor4.desc.dims), at::kFloat);
    // get ref output tensor
    atInRefMixedQuery =
        atInRefMixedQuery.view({atInRefMixedQuery.sizes()[0], atInRefMixedQuery.sizes()[1] * paramHeadNum,
                                atInRefMixedQuery.sizes()[2] / paramHeadNum});
    atInRefMixedQuery = atInRefMixedQuery.transpose(0, 1);  // {2, 1, 3}
     atInRefMixedKey = atInRefMixedKey.view({atInRefMixedKey.sizes()[0], atInRefMixedKey.sizes()[1] * paramHeadNum,
                                            atInRefMixedKey.sizes()[2] / paramHeadNum});
    atInRefMixedKey = atInRefMixedKey.permute({1, 2, 0});   // {2, 3, 2}
    atInRefMixedValue =
        atInRefMixedValue.view({atInRefMixedValue.sizes()[0], atInRefMixedValue.sizes()[1] * paramHeadNum,
                                atInRefMixedValue.sizes()[2] / paramHeadNum});
    atInRefMixedValue = atInRefMixedValue.transpose(0, 1);  // {2, 2, 3}
    double scal = 1 / sqrt(paramDk);
    at::Tensor attentionScores = atInRefMixedQuery.bmm(atInRefMixedKey).contiguous();    // {2, 1, 2}
    attentionScores = attentionScores.mul(scal);    // {2, 1, 2}
    attentionScores = attentionScores.view({attentionScores.sizes()[0] / paramHeadNum, paramHeadNum,
                                            attentionScores.sizes()[1], attentionScores.sizes()[2]});   // {2, 1, 1, 2}
    attentionScores = attentionScores.add(atInRefAttentionMask);    // {2, 1, 1, 2}
    attentionScores = attentionScores.view({attentionScores.sizes()[0] * attentionScores.sizes()[1],
                                            attentionScores.sizes()[2], attentionScores.sizes()[3]});   // {2, 1, 2}
    at::Tensor attention_probs = attentionScores.softmax(-1);    // {2, 1, 2}
    at::Tensor contextLayer = attention_probs.bmm(atInRefMixedValue);    // {2, 1, 3}
    contextLayer = contextLayer.transpose(0, 1).contiguous();   // {1, 2, 3}
    at::Tensor atOutRefTensor = contextLayer
                                       .view({contextLayer.sizes()[0], contextLayer.sizes()[1] / paramHeadNum,
                                              contextLayer.sizes()[2] * paramHeadNum})
                                       .contiguous();   // {1, 2, 3}
    // compare
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)atOutRefTensor.storage().data_ptr().get(); // golden
    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = atRefOutArray[i];
        float actual = atOutArray[i];
        ASD_LOG(INFO) << "expect: " << expect << "actual: " << actual;
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

/// @brief chatglm6b golden
/// @param context 
/// @return 
AsdOps::Status SelfAttentionChatglm6bGolden(const GoldenContext &context)
{
    // define param

    // get constructed input/output tensors
    const AsdOps::Tensor inTensor = context.hostInTensors.at(0);
    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // construct ref input tensors
    at::Tensor atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kFloat);
    // get ref output tensor
    at::Tensor atOutRefTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // compare
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)atOutRefTensor.storage().data_ptr().get(); // golden
    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = atRefOutArray[i];
        float actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

/// @brief chatglm2_6b golden
/// @param context 
/// @return 
AsdOps::Status SelfAttentionChatglm2_6bGolden(const GoldenContext &context)
{
    // define param

    // get constructed input/output tensors
    const AsdOps::Tensor inTensor = context.hostInTensors.at(0);
    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // construct ref input tensors
    at::Tensor atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kFloat);
    // get ref output tensor
    at::Tensor atOutRefTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // compare
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)atOutRefTensor.storage().data_ptr().get(); // golden
    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = atRefOutArray[i];
        float actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

/// @brief llama7b golden
/// @param context 
/// @return 
AsdOps::Status SelfAttentionLlama7bGolden(const GoldenContext &context)
{
    // define param

    // get constructed input/output tensors
    const AsdOps::Tensor inTensor1 = context.hostInTensors.at(0);
    const AsdOps::Tensor inTensor2 = context.hostInTensors.at(0);
    const AsdOps::Tensor inTensor3 = context.hostInTensors.at(0);
    const AsdOps::Tensor inTensor4 = context.hostInTensors.at(0);
    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // construct ref input tensors
    // at::Tensor atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kFloat);
    // get ref output tensor
    at::Tensor atOutRefTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // compare
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)atOutRefTensor.storage().data_ptr().get(); // golden
    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = atRefOutArray[i];
        float actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

/// @brief openbert test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, TestSelfAttentionOpenbert)
{
    AclTransformer::SelfAttentionParam param;
    param.dk = 1;
    param.headNum = 1;
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 2}}};
    OpTest opTest;
    opTest.Golden(&SelfAttentionOpenbertGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

/// @brief chatglm6b test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, TestSelfAttentionChatglm6b)
{
    AclTransformer::SelfAttentionParam param;
    param.model = "chatglm6b";
    // param.dk = 1;
    // param.headNum = 1;
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 2}}};
    OpTest opTest;
    opTest.Golden(&SelfAttentionChatglm6bGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

/// @brief chatglm2_6b test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, TestSelfAttentionChatglm2_6b)
{
    AclTransformer::SelfAttentionParam param;
    param.model = "chatglm2_6b";
    // param.dk = 1;
    // param.headNum = 1;
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 2, 3}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 2}}};
    OpTest opTest;
    opTest.Golden(&SelfAttentionChatglm2_6bGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

/// @brief llama7b test
/// @param  
/// @param  
TEST(TestSelfAttentionOperation, TestSelfAttentionLlama7b)
{
    AclTransformer::SelfAttentionParam param;
    param.model = "llama7b";
    // param.dk = 1;
    // param.headNum = 1;
    AclTransformer::SelfAttentionOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}}};
    OpTest opTest;
    opTest.Golden(&SelfAttentionLlama7bGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}