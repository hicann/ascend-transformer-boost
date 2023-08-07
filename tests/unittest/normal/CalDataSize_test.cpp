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
#include "acltransformer/utils/tensor_util.h"
#include <sstream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <asdops/utils/binfile/binfile.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/filesystem/filesystem.h>

using namespace AclTransformer;
using namespace AsdOps;

TEST(CalTest, CalcTensorDataSizeTest1){
    AsdOps::Tensor tensor;
    tensor.desc.dtype=TENSOR_DTYPE_FLOAT16;
    tensor.desc.dims={3, 4, 7};
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor),168);

}

TEST(CalTest, CalcTensorDataSizeTest2){
    AsdOps::Tensor tensor;
    tensor.desc.dtype=TENSOR_DTYPE_DOUBLE;
    tensor.desc.dims={3, 4, 7};
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor),0);

}

TEST(CalTest, CalcTensorDataSizeTest3){
    AsdOps::Tensor tensor;
    tensor.desc.dtype=TENSOR_DTYPE_FLOAT16;
    tensor.desc.dims={};
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor),0);

}