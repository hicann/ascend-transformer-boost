/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <securec.h>
#include <gtest/gtest.h>
#include <iostream>
#include <atb/utils/log.h>
#include <atb/core/runner_variant_pack.h>
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
 
using namespace atb;
using namespace Mki;

TEST(TestSVector, SVectorHeap)
{
    atb::SVector<atb::Tensor> inTensors;
    inTensors.reserve(100);

    atb::Tensor firstTensor;
    firstTensor.desc.dtype = ACL_INT16;
    firstTensor.desc.shape.dimNum = 2;
    firstTensor.desc.shape.dims[0] = 1;
    firstTensor.desc.shape.dims[1] = 1;

    atb::Tensor tensorItem;
    tensorItem.desc.dtype = ACL_INT16;
    tensorItem.desc.shape.dimNum = 2;
    tensorItem.desc.shape.dims[0] = 2;
    tensorItem.desc.shape.dims[1] = 5;

    atb::Tensor endTensor;
    endTensor.desc.dtype = ACL_INT16;
    endTensor.desc.shape.dimNum = 2;
    endTensor.desc.shape.dims[0] = 10;
    endTensor.desc.shape.dims[1] = 10;

    inTensors.insert(0, firstTensor);
    EXPECT_EQ(inTensors.at(0).desc.shape.dims[0], 1);
    
    for (int i = 1; i < 50; i++) {
        inTensors.insert(i, tensorItem);
    }

    inTensors.push_back(endTensor);
    EXPECT_EQ(inTensors.at(50).desc.shape.dims[0], 10);
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensors.size(), 51);

    EXPECT_EQ(inTensors.begin()->desc.shape.dims[0], 1);
    inTensors.clear();
    EXPECT_EQ(inTensors.size(), 0);
    EXPECT_EQ(inTensors.empty(), true);
}

TEST(TestSVector, HeapInitializerlist)
{
    atb::SVector<uint64_t> inTensors;
    inTensors.reserve(120);
    inTensors = {11, 1234, 12345, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100};
    EXPECT_EQ(inTensors.at(0), 11);

    inTensors.push_back(101);
    EXPECT_EQ(inTensors.at(100), 101);
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensors.size(), 101);
    inTensors.clear();
    EXPECT_EQ(inTensors.size(), 0);
    EXPECT_EQ(inTensors.empty(), true);
    
    atb::SVector<bool> inTensorPerms;
    inTensorPerms.reserve(133);
    inTensorPerms.resize(133);
    for (size_t i = 0; i < inTensorPerms.size(); ++i) {
        inTensorPerms.at(i) = false;
    }
    EXPECT_EQ(inTensorPerms.at(132), false);
}

TEST(TestSVector, HeapInitializerlistTwo)
{
    atb::SVector<bool> inTensorPerms;
    inTensorPerms.reserve(133);
    inTensorPerms.resize(133);
    for (size_t i = 0; i < inTensorPerms.size(); ++i) {
        inTensorPerms.at(i) = false;
    }
    EXPECT_EQ(inTensorPerms.at(132), false);
}

TEST(TestSVector, StackInitializerlist)
{
    atb::SVector<int16_t> inTensors;
    inTensors = {11, 1234, 12345, 1, 7, 100};
    EXPECT_EQ(inTensors.at(0), 11);
    EXPECT_EQ(inTensors.at(1), 1234);
    EXPECT_EQ(inTensors.at(2), 12345);

    inTensors.push_back(101);
    EXPECT_EQ(inTensors.at(6), 101);
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensors.size(), 7);
    inTensors.clear();
    EXPECT_EQ(inTensors.size(), 0);
    EXPECT_EQ(inTensors.empty(), true);
}

TEST(TestSVector, SVectorStack)
{
    atb::SVector<atb::Tensor> inTensors;
    atb::Tensor tensorItem;
    tensorItem.desc.dtype = ACL_INT16;
    tensorItem.desc.shape.dimNum = 2;
    tensorItem.desc.shape.dims[0] = 2;
    tensorItem.desc.shape.dims[1] = 5;

    atb::Tensor firstTensor;
    firstTensor.desc.dtype = ACL_INT16;
    firstTensor.desc.shape.dimNum = 2;
    firstTensor.desc.shape.dims[0] = 1;
    firstTensor.desc.shape.dims[1] = 1;

    atb::Tensor endTensor;
    endTensor.desc.dtype = ACL_INT16;
    endTensor.desc.shape.dimNum = 2;
    endTensor.desc.shape.dims[0] = 10;
    endTensor.desc.shape.dims[1] = 10;

    inTensors.insert(0, firstTensor);
    EXPECT_EQ(inTensors.at(0).desc.shape.dims[0], 1);
    
    for (int i = 1; i < 63; i++) {
        inTensors.insert(i, tensorItem);
    }

    inTensors.push_back(endTensor);
    EXPECT_EQ(inTensors.at(63).desc.shape.dims[0], 10);
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensors.size(), 64);
    EXPECT_EQ(inTensors.begin()->desc.shape.dims[0], 1);
    EXPECT_EQ(inTensors.size(), 64);
    inTensors.clear();
    EXPECT_EQ(inTensors.size(), 0);
    EXPECT_EQ(inTensors.empty(), true);
}

TEST(TestSVector, HeapCopyConstructor)
{
    atb::SVector<uint64_t> inTensorSrc;
    inTensorSrc.reserve(100);
    inTensorSrc = {11, 1234, 12345, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100};
    atb::SVector<uint64_t> inTensorDest;
    inTensorDest.reserve(100);
    inTensorDest.resize(100);
    inTensorDest = inTensorSrc;
    EXPECT_EQ(inTensorDest.at(0), 11);
    EXPECT_EQ(inTensorDest.at(1), 1234);
    EXPECT_EQ(inTensorDest.at(2), 12345);
    
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensorDest.size(), 100);
    inTensorDest.clear();
    EXPECT_EQ(inTensorDest.size(), 0);
    EXPECT_EQ(inTensorDest.empty(), true);
}