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
#include <gtest/stub.h>
#include <torch/torch.h>
#include <half.hpp>
#include <asdops/utils/log/log.h>
#include "tests/unittest/test_util/test_common.h"
#include "acltransformer/ops/ffn_operation.h"
#include "tests/unittest/test_util/operation_test.h"
#include "core/include/acltransformer/config.h"
#include <asdops/utils/rt/rt.h>

using namespace AclTransformer;
using namespace AsdOps;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

bool Is910BStub(void* obj){
    const int versionLen = 32;
    char version[versionLen] = {0};
    AsdRtDeviceGetSocVersion(version, versionLen);
    ASD_LOG(INFO) << "SocVersion:" << std::string(version);
    return std::string(version).find("Ascend910B") == std::string::npos;
}
AsdOps::Status FfnGolden(const GoldenContext &);
AsdOps::Status FfnWithoutBiasGolden(const GoldenContext &);


TEST(TestFfnOperationStub, InferShape)
{
    AclTransformer::FfnParam param;
    AclTransformer::FfnOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 2}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT16);
    EXPECT_EQ(outTensorDescs.at(0).format, AsdOps::TENSOR_FORMAT_ND);
    AsdOps::SVector<int64_t> expectDims = {1, 2, 1};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
}

TEST(TestFfnOperationStub, TestFfn)
{
    Stub stub;
    stub.set(ADDR(Config, Is910B), Is910BStub);
    AclTransformer::FfnParam param;
    AclTransformer::FfnOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}}};
    OperationTest opTest;
    opTest.Golden(&FfnGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    // TENSOR_FORMAT_FRACTAL_NZ
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescsNz = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_FRACTAL_NZ, {1, 1, 1}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}}};
    AsdOps::Status statusNz = opTest.Run(&op, inTensorDescsNz);
    
}

TEST(TestFfnOperationStub, TestFfnWithoutBias)
{
    Stub stub;
    stub.set(ADDR(Config, Is910B), Is910BStub);
    AclTransformer::FfnParam param;
    param.hasBias = 0;
    AclTransformer::FfnOperation op(param);
    AsdOps::SVector<AsdOps::TensorDesc> inTensorDescs = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}}};
    OperationTest opTest;
    opTest.Golden(&FfnWithoutBiasGolden);
    AsdOps::Status status = opTest.Run(&op, inTensorDescs);
    // TENSOR_FORMAT_FRACTAL_NZ
     AsdOps::SVector<AsdOps::TensorDesc> inTensorDescsNz = {
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1}},
        {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_FRACTAL_NZ, {1, 1, 1}}};
    AsdOps::Status statusNz = opTest.Run(&op, inTensorDescsNz);
}