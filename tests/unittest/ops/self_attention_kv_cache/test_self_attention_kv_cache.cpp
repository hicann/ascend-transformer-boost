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
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

/// @brief chatglm6b(default) infershape test
/// @param  
/// @param  
TEST(TestSelfAttentionKvCacheOperation, InferShapeChatglm6b)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    AclTransformer::SelfAttentionKvCacheOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {5, 6, 7, 8}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {6, 7, 8, 9}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<AsdOps::SVector<int64_t>> expectDims = {{1, 2, 12}, {6, 6, 7, 8}, {7, 7, 8, 9}};
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

/// @brief chatglm2_6b infershape test
/// @param  
/// @param  
TEST(TestSelfAttentionKvCacheOperation, InferShapeChatglm2_6b)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    param.model = "chatglm2_6b";
    AclTransformer::SelfAttentionKvCacheOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {5, 6, 7, 8}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<AsdOps::SVector<int64_t>> expectDims = {{1, 2, 12}, {5, 5, 6, 7}, {6, 6, 7, 8}};
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

/// @brief bloom7b infershape test
/// @param  
/// @param  
TEST(TestSelfAttentionKvCacheOperation, InferShapeBloom7b)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    param.model = "bloom7b";
    AclTransformer::SelfAttentionKvCacheOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {5, 6, 7, 8}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<AsdOps::SVector<int64_t>> expectDims = {{1, 2, 1, 4}, {2, 3, 5}, {3, 5, 5}};
    ASSERT_EQ(expectDims.at(0).size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0).at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(0).at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(0).at(2), outTensorDescs.at(0).dims.at(2));
    EXPECT_EQ(expectDims.at(0).at(3), outTensorDescs.at(0).dims.at(3));
    ASSERT_EQ(expectDims.at(1).size(), outTensorDescs.at(1).dims.size());
    EXPECT_EQ(expectDims.at(1).at(0), outTensorDescs.at(1).dims.at(0));
    EXPECT_EQ(expectDims.at(1).at(1), outTensorDescs.at(1).dims.at(1));
    EXPECT_EQ(expectDims.at(1).at(2), outTensorDescs.at(1).dims.at(2));
    ASSERT_EQ(expectDims.at(2).size(), outTensorDescs.at(2).dims.size());
    EXPECT_EQ(expectDims.at(2).at(0), outTensorDescs.at(2).dims.at(0));
    EXPECT_EQ(expectDims.at(2).at(1), outTensorDescs.at(2).dims.at(1));
    EXPECT_EQ(expectDims.at(2).at(2), outTensorDescs.at(2).dims.at(2));
}

/// @brief chatglm6b golden
/// @param context
/// @return
AsdOps::Status SelfAttentionKvCacheChatglm6bGolden(const GoldenContext &context)
{
    // define param
    int64_t paramHeadNum = 1;
    int64_t paramLayerId = 1;
    // get constructed input/output tensors
    const AsdOps::Tensor inTensor1 = context.hostInTensors.at(0);
    const AsdOps::Tensor inTensor2 = context.hostInTensors.at(1);
    const AsdOps::Tensor inTensor3 = context.hostInTensors.at(2);
    const AsdOps::Tensor inTensor4 = context.hostInTensors.at(3);
    const AsdOps::Tensor inTensor5 = context.hostInTensors.at(4);
    const AsdOps::Tensor inTensor6 = context.hostInTensors.at(5);
    const AsdOps::Tensor outTensor1 = context.hostOutTensors.at(0);
    const AsdOps::Tensor outTensor2 = context.hostOutTensors.at(1);
    const AsdOps::Tensor outTensor3 = context.hostOutTensors.at(2);
    at::Tensor atOutTensor = at::from_blob(outTensor1.data, ToIntArrayRef(outTensor1.desc.dims), at::kFloat);
    at::Tensor atOutTensor1 = at::from_blob(outTensor2.data, ToIntArrayRef(outTensor2.desc.dims), at::kFloat);
    at::Tensor atOutTensor2 = at::from_blob(outTensor3.data, ToIntArrayRef(outTensor3.desc.dims), at::kFloat);
    // construct ref input tensors
    at::Tensor atInRefMixedQuery = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
    at::Tensor atInRefMixedKey = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kFloat);
    at::Tensor atInRefMixedValue = at::from_blob(inTensor3.data, ToIntArrayRef(inTensor3.desc.dims), at::kFloat);
    at::Tensor atInRefAttentionMask = at::from_blob(inTensor4.data, ToIntArrayRef(inTensor4.desc.dims), at::kFloat);
    at::Tensor atInRefPastKey = at::from_blob(inTensor5.data, ToIntArrayRef(inTensor5.desc.dims), at::kFloat);
    at::Tensor atInRefPastValue = at::from_blob(inTensor6.data, ToIntArrayRef(inTensor6.desc.dims), at::kFloat);
    // get ref output tensor
    at::Tensor presentKey = torch::cat({atInRefPastKey, atInRefMixedKey}, 0).contiguous();
    // TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, presentKey, runnerVariantPack.outTensors[1]);
    atInRefMixedQuery = atInRefMixedQuery.view({atInRefMixedQuery.sizes()[0],
                                                atInRefMixedQuery.sizes()[1] * atInRefMixedQuery.sizes()[2],
                                                atInRefMixedQuery.sizes()[3]});
    atInRefMixedQuery = atInRefMixedQuery.transpose(0, 1);
    at::Tensor presentValue = torch::cat({atInRefPastValue, atInRefMixedValue}, 0).contiguous();
    // TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, presentValue, runnerVariantPack.outTensors[2]);
    presentValue = presentValue.view(
        {presentValue.sizes()[0], presentValue.sizes()[1] * presentValue.sizes()[2], presentValue.sizes()[3]});
    presentValue = presentValue.transpose(0, 1);
    presentKey =
        presentKey.view({presentKey.sizes()[0], presentKey.sizes()[1] * presentKey.sizes()[2], presentKey.sizes()[3]});
    presentKey = presentKey.permute({1, 2, 0});
    atInRefMixedQuery = atInRefMixedQuery / (sqrt(presentValue.sizes()[2]) * (float)(paramLayerId + 1));
    at::Tensor attentionScores = atInRefMixedQuery.bmm(presentKey).contiguous();
    attentionScores = attentionScores.view({attentionScores.sizes()[0] / paramHeadNum, paramHeadNum,
                                            attentionScores.sizes()[1], attentionScores.sizes()[2]});
    if (atInRefAttentionMask.sum().item<int64_t>() > 0) {
        attentionScores.masked_fill_(atInRefAttentionMask, -10000.0);
    }
    attentionScores = attentionScores.to(torch::kFloat32);
    attentionScores = attentionScores.mul(paramLayerId + 1.0);
    at::Tensor attention_probs = attentionScores.softmax(-1);
    attention_probs = attention_probs.to(torch::kHalf);
    attention_probs = attention_probs.view({attentionScores.sizes()[0] * attentionScores.sizes()[1],
                                            attentionScores.sizes()[2], attentionScores.sizes()[3]});
    at::Tensor contextLayer = attention_probs.bmm(presentValue);
    contextLayer = contextLayer.transpose(0, 1);
    contextLayer = contextLayer
                       .view({contextLayer.sizes()[0], contextLayer.sizes()[1] / paramHeadNum, paramHeadNum,
                              contextLayer.sizes()[2]})
                       .contiguous();
    contextLayer =
        contextLayer
            .view({contextLayer.sizes()[0], contextLayer.sizes()[1], contextLayer.sizes()[2] * contextLayer.sizes()[3]})
            .contiguous();
    // compare
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)contextLayer.storage().data_ptr().get(); // golden
    for (int i = 0; i < outTensor1.Numel(); i++) {
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
AsdOps::Status SelfAttentionKvCacheLlama7bGolden(const GoldenContext &context)
{
    // define param
    int64_t paramHeadNum = 1;
    int64_t paramLayerId = 1;
    int64_t paramDk = 1;
    // get constructed input/output tensors
    const AsdOps::Tensor inTensor1 = context.hostInTensors.at(0);
    const AsdOps::Tensor inTensor2 = context.hostInTensors.at(1);
    const AsdOps::Tensor inTensor3 = context.hostInTensors.at(2);
    const AsdOps::Tensor inTensor4 = context.hostInTensors.at(3);
    const AsdOps::Tensor inTensor5 = context.hostInTensors.at(4);
    const AsdOps::Tensor inTensor6 = context.hostInTensors.at(5);
    const AsdOps::Tensor outTensor1 = context.hostOutTensors.at(0);
    const AsdOps::Tensor outTensor2 = context.hostOutTensors.at(1);
    const AsdOps::Tensor outTensor3 = context.hostOutTensors.at(2);
    at::Tensor atOutTensor = at::from_blob(outTensor1.data, ToIntArrayRef(outTensor1.desc.dims), at::kFloat);
    // construct ref input tensors
    at::Tensor atInRefMixedQuery = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
    at::Tensor atInRefMixedKey = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kFloat);
    at::Tensor atInRefMixedValue = at::from_blob(inTensor3.data, ToIntArrayRef(inTensor3.desc.dims), at::kFloat);
    at::Tensor atInRefAttentionMask = at::from_blob(inTensor4.data, ToIntArrayRef(inTensor4.desc.dims), at::kFloat);
    at::Tensor atInRefPastKey = at::from_blob(inTensor5.data, ToIntArrayRef(inTensor5.desc.dims), at::kFloat);
    at::Tensor atInRefPastValue = at::from_blob(inTensor6.data, ToIntArrayRef(inTensor6.desc.dims), at::kFloat);
    // get ref output tensor
    atInRefMixedQuery = atInRefMixedQuery.permute({1, 2, 0, 3});
    atInRefMixedKey = atInRefMixedKey.permute({1, 2, 0, 3});
    atInRefMixedValue = atInRefMixedValue.permute({1, 2, 0, 3});
    atInRefPastKey = atInRefPastKey.permute({1, 2, 0, 3});
    atInRefPastValue = atInRefPastValue.permute({1, 2, 0, 3});
    at::Tensor presentKey = torch::cat({atInRefPastKey, atInRefMixedKey}, 0).contiguous();
    at::Tensor presentValue = torch::cat({atInRefPastValue, atInRefMixedValue}, 0).contiguous();
    at::Tensor presentKeyOut = presentKey;
    presentKeyOut = presentKeyOut.permute({2, 0, 1, 3}).contiguous();
    // torch::save(presentKeyOut.to(at::Device(at::kCPU)), "presentKeyOut_example.path");
    // TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, presentKeyOut, runnerVariantPack.outTensors[1]);
    at::Tensor presentValueOut = presentValue;
    presentValueOut = presentValueOut.permute({2, 0, 1, 3}).contiguous();
    // TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, presentValueOut, runnerVariantPack.outTensors[2]);
    presentKey = presentKey.transpose(2, 3);
    atInRefMixedQuery = atInRefMixedQuery.view({atInRefMixedQuery.sizes()[0] * atInRefMixedQuery.sizes()[1],
                                                atInRefMixedQuery.sizes()[2], atInRefMixedQuery.sizes()[3]});
    presentKey =
        presentKey.view({presentKey.sizes()[0] * presentKey.sizes()[1], presentKey.sizes()[2], presentKey.sizes()[3]});
    at::Tensor attentionScores = atInRefMixedQuery.bmm(presentKey);
    attentionScores = attentionScores / sqrt(paramDk);
    atInRefAttentionMask =
        atInRefAttentionMask.view({atInRefAttentionMask.sizes()[0] * atInRefAttentionMask.sizes()[1],
                                   atInRefAttentionMask.sizes()[2], atInRefAttentionMask.sizes()[3]});
    attentionScores = attentionScores + atInRefAttentionMask;
    attentionScores = attentionScores.to(torch::kFloat32);
    at::Tensor attention_probs = attentionScores.softmax(-1);
    attention_probs = attention_probs.to(torch::kHalf);
    presentValue = presentValue.view(
        {presentValue.sizes()[0] * presentValue.sizes()[1], presentValue.sizes()[2], presentValue.sizes()[3]});
    at::Tensor contextLayer = attention_probs.bmm(presentValue);
    contextLayer = contextLayer.view(
        {contextLayer.sizes()[0] / paramHeadNum, paramHeadNum, contextLayer.sizes()[1], contextLayer.sizes()[2]});
    contextLayer = contextLayer.transpose(1, 2);
    contextLayer = contextLayer.view(
        {contextLayer.sizes()[0], contextLayer.sizes()[1], contextLayer.sizes()[2] * contextLayer.sizes()[3]});
    contextLayer = contextLayer.transpose(0, 1).contiguous();
    // compare
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)contextLayer.storage().data_ptr().get(); // golden
    for (int i = 0; i < outTensor1.Numel(); i++) {
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
AsdOps::Status SelfAttentionKvCacheChatglm2_6bGolden(const GoldenContext &context) 
{
    // define param

    // get constructed input/output tensors
    const AsdOps::Tensor inTensor1 = context.hostInTensors.at(0);
    const AsdOps::Tensor inTensor2 = context.hostInTensors.at(1);
    const AsdOps::Tensor inTensor3 = context.hostInTensors.at(2);
    const AsdOps::Tensor inTensor4 = context.hostInTensors.at(3);
    const AsdOps::Tensor inTensor5 = context.hostInTensors.at(4);
    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    const AsdOps::Tensor outTensor2 = context.hostOutTensors.at(1);
    const AsdOps::Tensor outTensor3 = context.hostOutTensors.at(2);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // construct ref input tensors
    at::Tensor atInRefTensor = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
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

/// @brief bloom7b golden
/// @param context 
/// @return 
AsdOps::Status SelfAttentionKvCacheBloom7bGolden(const GoldenContext &context) 
{
    // define param

    // get constructed input/output tensors
    const AsdOps::Tensor inTensor1 = context.hostInTensors.at(0);
    const AsdOps::Tensor inTensor2 = context.hostInTensors.at(1);
    const AsdOps::Tensor inTensor3 = context.hostInTensors.at(2);
    const AsdOps::Tensor inTensor4 = context.hostInTensors.at(3);
    const AsdOps::Tensor inTensor5 = context.hostInTensors.at(4);
    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    const AsdOps::Tensor outTensor2 = context.hostOutTensors.at(1);
    const AsdOps::Tensor outTensor3 = context.hostOutTensors.at(2);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // construct ref input tensors
    at::Tensor atInRefTensor = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
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
/// @brief chatglm6b test
/// @param  
/// @param  
TEST(TestSelfAttentionKvCacheOperation, TestSelfAttentionKvCacheChatglm6b)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    param.headNum = 1;
    param.layerId = 1;
    AclTransformer::SelfAttentionKvCacheOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {5, 6, 7, 8}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {6, 7, 8, 9}}};
    OpTest opTest;
    opTest.Golden(&SelfAttentionKvCacheChatglm6bGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    // ASSERT_EQ(status.Ok(), true);
}

/// @brief llama7b test
/// @param  
/// @param  
TEST(TestSelfAttentionKvCacheOperation, TestSelfAttentionKvCacheLlama7b)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    param.model = "llama7b";
    param.headNum = 1;
    param.layerId = 1;
    param.dk = 1;
    AclTransformer::SelfAttentionKvCacheOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {5, 6, 7, 8}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {6, 7, 8, 9}}};
    OpTest opTest;
    opTest.Golden(&SelfAttentionKvCacheLlama7bGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    // ASSERT_EQ(status.Ok(), true);
}

/// @brief chatglm2_6b test
/// @param  
/// @param  
TEST(TestSelfAttentionKvCacheOperation, TestSelfAttentionKvCacheChatglm2_6b)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    param.model = "chatglm2_6b";
    AclTransformer::SelfAttentionKvCacheOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {5, 6, 7, 8}}};
    OpTest opTest;
    opTest.Golden(&SelfAttentionKvCacheChatglm2_6bGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    // ASSERT_EQ(status.Ok(), true);
}

/// @brief bloom7b test
/// @param  
/// @param  
TEST(TestSelfAttentionKvCacheOperation, TestSelfAttentionKvCacheBloom7b)
{
    AclTransformer::SelfAttentionKvCacheParam param;
    param.model = "bloom7b";
    AclTransformer::SelfAttentionKvCacheOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2, 3, 4}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {2, 3, 4, 5}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {3, 4, 5, 6}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {4, 5, 6, 7}},
        {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {5, 6, 7, 8}}};
    OpTest opTest;
    opTest.Golden(&SelfAttentionKvCacheBloom7bGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    // ASSERT_EQ(status.Ok(), true);
}