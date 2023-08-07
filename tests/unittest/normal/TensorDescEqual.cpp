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
#include <vector>
#include <string>
#include <string.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include "acltransformer/utils/tensor_util.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "asdops/utils/filesystem/filesystem.h"
using namespace AclTransformer;
using namespace AsdOps;

TEST(TestTensorUtil,TensorDescNOTEqualTest){
    AsdOps::TensorDesc TensorDescA;
    TensorDescA.dtype=TENSOR_DTYPE_INT32;
    TensorDescA.format=TENSOR_FORMAT_ND;
    SVector<int64_t> dimsA = {3, 5};
    AsdOps::TensorDesc TensorDescB;
    TensorDescB.dtype=TENSOR_DTYPE_INT32;
    TensorDescB.format=TENSOR_FORMAT_ND;
    SVector<int64_t> dimsB = {4, 6};
    EXPECT_EQ(TensorUtil::AsdOpsTensorDescEqual(TensorDescA,TensorDescB),false)
}

TEST(TestTensorUtil,TensorDescEqualTest){
    AsdOps::TensorDesc TensorDescA;
    TensorDescA.dtype=TENSOR_DTYPE_INT32;
    TensorDescA.format=TENSOR_FORMAT_ND;
    SVector<int64_t> dimsA = {2, 5};
    AsdOps::TensorDesc TensorDescB;
    TensorDescB.dtype=TENSOR_DTYPE_INT32;
    TensorDescB.format=TENSOR_FORMAT_ND;
    SVector<int64_t> dimsB = {2, 5};
    EXPECT_EQ(TensorUtil::AsdOpsTensorDescEqual(TensorDescA,TensorDescB),true)
}
