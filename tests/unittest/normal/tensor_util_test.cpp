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

TEST(TestTensorUtil, LoadTensorTest)
{
    const int currentDevId = 2;
    const int dataCount = 10;
    const int dataSize = dataCount * sizeof(int32_t);
    int32_t data[dataCount] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const char *envStr = std::getenv("ACLTRANSFORMER_HOME_PATH");
    ASSERT_NE(envStr, nullptr);
    std::string pathStr(envStr);
    std::string dirStr = pathStr + "/testdata";
    pathStr = pathStr + "/testdata/tensor1";

    if (!FileSystem::IsDir(dirStr)) {
        FileSystem::MakeDir(dirStr, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    
    void *dataPtr = static_cast<void*>(data);
    std::vector<char> hostData(dataSize);
    memcpy(hostData.data(), static_cast<char*>(dataPtr), dataSize);

    AsdOps::Tensor tensor;
    tensor.desc.dtype = TensorDType::TENSOR_DTYPE_INT32;
    tensor.desc.format = TensorFormat::TENSOR_FORMAT_ND;
    SVector<int64_t> dims = {2, 5};
    tensor.desc.dims = dims;
    tensor.dataSize = dataSize;
    int ret = AsdRtDeviceSetCurrent(currentDevId);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;
    ret = AsdRtMemMallocDevice(&tensor.data, dataSize, ASDRT_MEM_DEFAULT);
    ASSERT_EQ(ret, ASDRT_SUCCESS);
    AsdRtMemCopy(tensor.data, dataSize, hostData.data(), dataSize, ASDRT_MEMCOPY_HOST_TO_DEVICE);
    TensorUtil::SaveTensor(tensor, pathStr);

    AsdOps::Tensor tensor1;
    TensorUtil::LoadTensor(tensor1, pathStr);
    EXPECT_EQ(tensor1.desc.dtype, TensorDType::TENSOR_DTYPE_INT32);
    EXPECT_EQ(tensor1.desc.format, TensorFormat::TENSOR_FORMAT_ND);
    EXPECT_EQ(tensor1.desc.dims.at(0), 2);
    EXPECT_EQ(tensor1.desc.dims.at(1), 5);
    EXPECT_EQ(tensor1.dataSize, dataSize);
    std::vector<char> hostData1(dataSize);
    AsdRtMemCopy(hostData1.data(), dataSize, tensor.data, dataSize, ASDRT_MEMCOPY_DEVICE_TO_HOST);
    EXPECT_EQ(hostData, hostData1);
}