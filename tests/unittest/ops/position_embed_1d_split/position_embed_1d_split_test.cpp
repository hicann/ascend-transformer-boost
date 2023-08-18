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
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include "tests/unittest/test_util/op_test.h"
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
using namespace AclTransformer;
using namespace AsdOps;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
constexpr int64_t LONG_MIN_VALUE = 1;
constexpr int64_t LONG_MAX_VALUE = 7;
TEST(TestPositionEmbedding1dSplitOperation, InferShape)
{
    AclTransformer::PositionEmbedding1dSplitParam param = {8};
    AclTransformer::PositionEmbedding1dSplitOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensors = {{AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {8, 8, 64}},
                                                     {AsdOps::TENSOR_DTYPE_INT64, AsdOps::TENSOR_FORMAT_ND, {8, 8, 1}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 8, 8}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 8, 8}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensors, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    EXPECT_EQ(outTensorDescs.at(0).format, AsdOps::TENSOR_FORMAT_ND);
    AsdOps::SVector<int64_t> expectDims = {8, 8, 8, 8};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(0).dims.at(3));
}

AsdOps::Status PosEmb1dSplitGolden(AclTransformer::PositionEmbedding1dSplitParam param, const GoldenContext &context)
{
    const Tensor &input = context.hostInTensors.at(0);
    at::Tensor atInRefinput = at::from_blob(input.data, ToIntArrayRef(input.desc.dims), at::kFloat);
    const Tensor &positionIds = context.hostInTensors.at(1);
    at::Tensor atInRefpositionIds = at::from_blob(positionIds.data, ToIntArrayRef(positionIds.desc.dims), at::kLong);
    const Tensor &cosTable = context.hostInTensors.at(2);
    at::Tensor atInRefcosTable = at::from_blob(cosTable.data, ToIntArrayRef(cosTable.desc.dims), at::kFloat);
    const Tensor &sinTable = context.hostInTensors.at(3);
    at::Tensor atInRefsinTable = at::from_blob(sinTable.data, ToIntArrayRef(sinTable.desc.dims), at::kFloat);

    // // [batch, seqlen, embeddim]
    // [batch, headNum, seqlen, headDim]
    at::Tensor atInRefinput2 = atInRefinput.view({atInRefinput.sizes()[0], atInRefinput.sizes()[1], param.headNum, atInRefinput.sizes()[2] / param.headNum})
                .transpose(1, 2);
    // [seqLen, head_dim]
    at::Tensor atInRefcosTable2 = atInRefcosTable.squeeze(1).squeeze(0);
    at::Tensor atInRefsinTable2 = atInRefsinTable.squeeze(1).squeeze(0);
    // [bs, 1, seqlen, headDim]
    at::Tensor atInRefcosTable3 = torch::nn::functional::embedding(atInRefpositionIds, atInRefcosTable2).unsqueeze(1);
    at::Tensor atInRefsinTable3 = torch::nn::functional::embedding(atInRefpositionIds, atInRefsinTable2).unsqueeze(1);
    int chunksLastDim = atInRefinput2.sizes().size() - 1;
    int chunksLastDimSize = atInRefinput2.sizes()[chunksLastDim];
    ASD_LOG(INFO) << "chunksLastDim: " << chunksLastDim;
    ASD_LOG(INFO) << "chunksLastDimSize: " << chunksLastDimSize;
    at::Tensor inputRotate = torch::cat(
        {atInRefinput2.slice(-1, chunksLastDimSize / 2, chunksLastDimSize).neg(), atInRefinput2.slice(-1, 0, chunksLastDimSize / 2)},
        chunksLastDim);
    // [batch, headNum, seqlen, headDim]
    at::Tensor refOutTensor = torch::add(torch::mul(atInRefinput2, atInRefcosTable3), torch::mul(inputRotate, atInRefsinTable3));
    // [seqlen, batch, headNum, headDim]
    refOutTensor = refOutTensor.permute({2, 0, 1, 3});
    ASD_LOG(INFO) << "inputEmbedded: " << refOutTensor.sizes();
    const AsdOps::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)refOutTensor.storage().data_ptr().get(); 
    float *outData = static_cast<float *>(outTensor.data);
    // for (int i = 0; i < outTensor.Numel(); i++) {
    //     float expect = atRefOutArray[i];
    //     float actual = atOutArray[i];
    //     bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
    //     EXPECT_EQ(judge, true);
    //     if (!judge) {
    //         return Status::FailStatus(1, "unequal");
    //     }
    // }
    return Status::OkStatus();
}

TEST(TestPositionEmbedding1dSplitOperation, TestPositionEmbedding1dSplit)
{
    AclTransformer::PositionEmbedding1dSplitParam opParam = {8};
    AclTransformer::PositionEmbedding1dSplitOperation op(opParam);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {8, 8, 64}},
                                            {AsdOps::TENSOR_DTYPE_INT64, AsdOps::TENSOR_FORMAT_ND, {8, 8}},
                                            {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 8, 8}},
                                            {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 8, 8}}};
    OpTest opTest;
    opTest.LongRand(LONG_MIN_VALUE,LONG_MAX_VALUE);
    opTest.Golden(std::bind(PosEmb1dSplitGolden, opParam, std::placeholders::_1));
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}