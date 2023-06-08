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
#include "acltransformer/ops/add_operation.h"

using namespace AclTransformer;
using namespace AsdOps;

TEST(TestAddOperation, InferShape)
{
    AclTransformer::AddParam param;
    AclTransformer::AddOperation op(param);
    std::vector<AsdOps::TensorDesc> inTensorDescs = {{AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2}},
                                                     {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 2}}};
    std::vector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 1);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    AsdOps::SVector<int64_t> expectDims = {1, 2};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
}