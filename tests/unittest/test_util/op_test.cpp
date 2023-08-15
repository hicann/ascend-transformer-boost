/**
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

#include "op_test.h"
#include <half.hpp>
#include <random>
#include <asdops/types.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/context/context.h"

namespace AclTransformer {
OpTest::OpTest()
{
    const char *envStr = std::getenv("SET_NPU_DEVICE");
    deviceId_ = (envStr != nullptr) ? atoi(envStr) : 0;
}

OpTest::~OpTest() { Cleanup(); }

void OpTest::Cleanup()
{
    for (auto tensor : goldenContext_.hostInTensors) {
        if (tensor.data) {
            free(tensor.data);
        }
    }
    goldenContext_.hostInTensors.clear();

    for (auto tensor : goldenContext_.hostOutTensors) {
        if (tensor.data) {
            free(tensor.data);
        }
    }
    goldenContext_.hostOutTensors.clear();

    for (auto tensor : goldenContext_.deviceInTensors) {
        if (tensor.data) {
            AsdRtMemFreeDevice(tensor.data);
        }
    }
    goldenContext_.deviceInTensors.clear();

    for (auto tensor : goldenContext_.deviceOutTensors) {
        if (tensor.data) {
            AsdRtMemFreeDevice(tensor.data);
        }
    }
    goldenContext_.deviceOutTensors.clear();

    if (handle_.stream) {
        AsdRtStreamDestroy(handle_.stream);
    }
}

void OpTest::Init()
{
    ASD_LOG(INFO) << "AsdRtDeviceSetCurrent " << deviceId_;
    int ret = AsdRtDeviceSetCurrent(deviceId_);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail";

    AsdRtStream stream = nullptr;
    ASD_LOG(INFO) << "AsdRtStreamCreate call";
    ret = AsdRtStreamCreate(&stream, 0);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtStreamCreate fail";

    handle_ = {stream};
}

AsdOps::Status OpTest::Prepare(AclTransformer::Operation *operation,
                               const AsdOps::SVector<AsdOps::Tensor> &inTensorLists)
{
    operation_ = operation;

    if (inTensorLists.size() != operation_->GetInTensorCount()) {
        return AsdOps::Status::FailStatus(-1, "InTensors Count not equal " +
                                                  std::to_string(operation_->GetInTensorCount()));
    }

    AsdOps::Status st = operation_->BuildPlan(&plan_);
    if (!st.Ok()) {
        return AsdOps::Status::FailStatus(-1, operation_->GetName() + " build plan fail");
    }

    return AsdOps::Status::OkStatus();
}

AsdOps::Tensor OpTest::CreateHostTensor(const AsdOps::Tensor &tensorIn)
{
    AsdOps::Tensor tensor;
    tensor.desc = tensorIn.desc;
    tensor.dataSize = tensorIn.dataSize;

    tensor.data = malloc(tensor.dataSize);
    memcpy(tensor.data, tensorIn.data, tensor.dataSize);

    return tensor;
}

AsdOps::Tensor OpTest::HostTensor2DeviceTensor(const AsdOps::Tensor &hostTensor)
{
    AsdOps::Tensor deviceTensor;
    deviceTensor.desc = hostTensor.desc;
    deviceTensor.dataSize = hostTensor.dataSize;
    int st = AsdRtMemMallocDevice(&deviceTensor.data, deviceTensor.dataSize, ASDRT_MEM_DEFAULT);
    if (st != 0) {
        ASD_LOG(ERROR) << "malloc device tensor fail";
        return deviceTensor;
    }
    st = AsdRtMemCopy(deviceTensor.data, deviceTensor.dataSize, hostTensor.data, hostTensor.dataSize,
                      ASDRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != 0) {
        ASD_LOG(ERROR) << "copy host tensor to device tensor";
    }
    return deviceTensor;
}

std::string OpTest::TensorToString(const AsdOps::Tensor &tensor)
{
    const int64_t printMaxCount = 10;
    std::ostringstream ss;
    ss << "dtype:" << tensor.desc.dtype << ", format:" << tensor.desc.format << ", numel:" << tensor.Numel()
       << ", dataSize:" << tensor.dataSize << ", data:[";

    if (tensor.data) {
        for (int64_t i = 0; i < tensor.Numel(); ++i) {
            if (i == printMaxCount) {
                ss << "...";
                break;
            }

            if (tensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT16) {
                half_float::half *tensorData = static_cast<half_float::half *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT) {
                float *tensorData = static_cast<float *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == AsdOps::TENSOR_DTYPE_INT32) {
                int32_t *tensorData = static_cast<int32_t *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == AsdOps::TENSOR_DTYPE_INT64) {
                int64_t *tensorData = static_cast<int64_t *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == AsdOps::TENSOR_DTYPE_INT8) {
                int8_t *tensorData = static_cast<int8_t *>(tensor.data);
                ss << static_cast<int>(tensorData[i]) << ",";
            } else if (tensor.desc.dtype == AsdOps::TENSOR_DTYPE_UINT32) {
                uint32_t *tensorData = static_cast<uint32_t *>(tensor.data);
                ss << tensorData[i] << ",";
            } else {
                ss << "N,";
            }
        }
    } else {
        ss << "null";
    }

    ss << "]";
    return ss.str();
}

AsdOps::Tensor OpTest::CreateHostZeroTensor(const AsdOps::TensorDesc &tensorDesc)
{
    AsdOps::Tensor tensor;
    tensor.desc = tensorDesc;

    std::random_device rd;
    std::default_random_engine eng(rd());
    if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_FLOAT || tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT32) {
        tensor.dataSize = tensor.Numel() * sizeof(float);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_FLOAT16) {
        tensor.dataSize = tensor.Numel() * sizeof(half_float::half);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT64) {
        tensor.dataSize = tensor.Numel() * sizeof(int64_t);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT32) {
        tensor.dataSize = tensor.Numel() * sizeof(int32_t);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_UINT32) {
        tensor.dataSize = tensor.Numel() * sizeof(uint32_t);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT8) {
        tensor.dataSize = tensor.Numel() * sizeof(int8_t);
    } else {
        ASD_LOG(ERROR) << "not support";
        return tensor;
    }

    tensor.data = malloc(tensor.dataSize);
    memset(tensor.data, 0, tensor.dataSize);
    return tensor;
}

void OpTest::BuildVariantPack(const AsdOps::SVector<AsdOps::Tensor> &inTensorLists)
{
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    outTensorDescs.resize(operation_->GetOutTensorCount());
    AsdOps::Status st = operation_->InferShape(inTensorLists, outTensorDescs);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "InferShape fail, error:" << st.Message();
        return;
    }

    variantPack_.inTensors.resize(operation_->GetInTensorCount());
    variantPack_.outTensors.resize(operation_->GetOutTensorCount());

    for (size_t i = 0; i < inTensorLists.size(); ++i) {
        AsdOps::Tensor &deviceTensor = variantPack_.inTensors.at(i);
        AsdOps::Tensor hostTensor = CreateHostTensor(inTensorLists.at(i));
        deviceTensor.dataSize = hostTensor.dataSize;
        deviceTensor = HostTensor2DeviceTensor(hostTensor);
        ASD_LOG(DEBUG) << "InTensor[" << i << "] = " << TensorToString(hostTensor);
        goldenContext_.hostInTensors.push_back(hostTensor);
        goldenContext_.deviceInTensors.push_back(deviceTensor);
    }

    for (size_t i = 0; i < variantPack_.outTensors.size(); ++i) {
        variantPack_.outTensors.at(i).desc = outTensorDescs.at(i);
        AsdOps::Tensor &deviceTensor = variantPack_.outTensors.at(i);
        AsdOps::Tensor hostTensor = CreateHostZeroTensor(deviceTensor.desc);
        deviceTensor.dataSize = hostTensor.dataSize;
        int st = AsdRtMemMallocDevice(&deviceTensor.data, deviceTensor.dataSize, ASDRT_MEM_DEFAULT);
        ASD_LOG_IF(st != 0, ERROR) << "malloc device tensor fail";
        goldenContext_.hostOutTensors.push_back(hostTensor);
        goldenContext_.deviceOutTensors.push_back(deviceTensor);
    }
}

AsdOps::Status OpTest::RunOperation()
{
    AsdOps::Status st = plan_.Setup(handle_, variantPack_);
    if (!st.Ok()) {
        return AsdOps::Status::FailStatus(-1, "Plan SetUp fail, error:" + st.Message());
    }

    variantPack_.workspaceSize = plan_.GetWorkspaceSize();
    ASD_LOG(INFO) << "get plan workspace size:" << variantPack_.workspaceSize;

    if (variantPack_.workspaceSize > 0) {
        variantPack_.workspace =
            AsdOps::GetSingleton<AclTransformer::Context>().GetWorkspaceBuffer(variantPack_.workspaceSize);
    }

    st = plan_.Execute(handle_, variantPack_);
    if (!st.Ok()) {
        return AsdOps::Status::FailStatus(-1, "Plan Execute fail, error:" + st.Message());
    }

    return AsdOps::Status::OkStatus();
}

AsdOps::Status OpTest::CopyDeviceTensorToHostTensor()
{
    for (size_t i = 0; i < variantPack_.outTensors.size(); ++i) {
        AsdOps::Tensor &deivceTensor = variantPack_.outTensors.at(i);
        AsdOps::Tensor &hostTensor = goldenContext_.hostOutTensors.at(i);
        ASD_LOG(DEBUG) << "AsdRtMemCopy start, hostTensor.data:" << hostTensor.data
                       << ", hostTensor.dataSize:" << hostTensor.dataSize << ", deivceTensor.data:" << deivceTensor.data
                       << ", deivceTensor.dataSize:" << deivceTensor.dataSize;
        int st = AsdRtMemCopy(hostTensor.data, hostTensor.dataSize, deivceTensor.data, deivceTensor.dataSize,
                              ASDRT_MEMCOPY_DEVICE_TO_HOST);
        if (st != 0) {
            ASD_LOG(ERROR) << "copy memory from device to host fail";
            return AsdOps::Status::FailStatus(-1, "copy memory from device to host fail");
        }
        ASD_LOG(DEBUG) << "OutTensor[" << i << "] = " << TensorToString(hostTensor);
    }
    return AsdOps::Status::OkStatus();
}

AsdOps::Status OpTest::Run(AclTransformer::Operation *operation,
                           const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs)
{
    AsdOps::SVector<AsdOps::Tensor> inTensors;
    inTensors.resize(inTensorDescs.size());
    GenerateRandomTensors(inTensorDescs, inTensors);
    return RunImpl(operation, inTensors);
}

AsdOps::Status OpTest::Run(AclTransformer::Operation *operation,
                           const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs,
                           const AsdOps::Any &varaintPackParam)
{
    AsdOps::SVector<AsdOps::Tensor> inTensors;
    inTensors.resize(inTensorDescs.size());
    GenerateRandomTensors(inTensorDescs, inTensors);
    return RunImpl(operation, inTensors, varaintPackParam);
}

AsdOps::Status OpTest::Run(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::Tensor> &inTensorLists)
{
    return RunImpl(operation, inTensorLists);
}

AsdOps::Status OpTest::Run(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::Tensor> &inTensorLists,
                           const AsdOps::Any &varaintPackParam)
{
    return RunImpl(operation, inTensorLists, varaintPackParam);
}

AsdOps::Status OpTest::RunImpl(AclTransformer::Operation *operation,
                               const AsdOps::SVector<AsdOps::Tensor> &inTensorLists,
                               const AsdOps::Any &varaintPackParam)
{
    Cleanup();
    Init();

    AsdOps::Status status = Prepare(operation, inTensorLists);
    if (!status.Ok()) {
        return status;
    }

    variantPack_.param = varaintPackParam;
    BuildVariantPack(inTensorLists);

    status = RunOperation();
    if (!status.Ok()) {
        return status;
    }

    status = CopyDeviceTensorToHostTensor();
    if (!status.Ok()) {
        return status;
    }

    status = RunGolden();
    if (!status.Ok()) {
        ASD_LOG(ERROR) << "golden fail, error:" << status.ToString();
        return status;
    }
    return AsdOps::Status::OkStatus();
}

AsdOps::Status OpTest::RunImpl(AclTransformer::Operation *operation,
                               const AsdOps::SVector<AsdOps::Tensor> &inTensorLists)
{
    Cleanup();
    Init();

    AsdOps::Status status = Prepare(operation, inTensorLists);
    if (!status.Ok()) {
        return status;
    }

    BuildVariantPack(inTensorLists);

    status = RunOperation();
    if (!status.Ok()) {
        return status;
    }

    status = CopyDeviceTensorToHostTensor();
    if (!status.Ok()) {
        return status;
    }

    status = RunGolden();
    if (!status.Ok()) {
        ASD_LOG(ERROR) << "golden fail, error:" << status.ToString();
        return status;
    }
    return AsdOps::Status::OkStatus();
}

AsdOps::Status OpTest::RunGolden()
{
    if (golden_) {
        return golden_(goldenContext_);
    }
    return AsdOps::Status::OkStatus();
}

AsdOps::Tensor OpTest::CreateHostRandTensor(const AsdOps::TensorDesc &tensorDesc)
{
    AsdOps::Tensor tensor;
    tensor.desc = tensorDesc;

    std::random_device rd;
    std::default_random_engine eng(rd());
    if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_FLOAT) {
        tensor.dataSize = tensor.Numel() * sizeof(float);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(randFloatMin_, randFloatMax_);
        float *tensorData = static_cast<float *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<float>(distr(eng));
        }
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_FLOAT16) {
        tensor.dataSize = tensor.Numel() * sizeof(half_float::half);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(randFloatMin_, randFloatMax_);
        half_float::half *tensorData = static_cast<half_float::half *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<half_float::half>(distr(eng));
        }
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT32) {
        tensor.dataSize = tensor.Numel() * sizeof(int32_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_int_distribution<int32_t> distr(randIntMin_, randIntMax_);
        int32_t *tensorData = static_cast<int32_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<int32_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT64) {
        tensor.dataSize = tensor.Numel() * sizeof(int64_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_int_distribution<int64_t> distr(randLongMin_, randLongMax_);
        int64_t *tensorData = static_cast<int64_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<int64_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_UINT32) {
        tensor.dataSize = tensor.Numel() * sizeof(uint32_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_int_distribution<uint32_t> distr(randLongMin_, randLongMax_);
        uint32_t *tensorData = static_cast<uint32_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<uint32_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT8) {
        tensor.dataSize = tensor.Numel() * sizeof(int8_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_int_distribution<int8_t> distr(randInt8Min_, randInt8Max_);
        int8_t *tensorData = static_cast<int8_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<int8_t>(distr(eng));
        }
    } else {
        ASD_LOG(ERROR) << "dtype not support in CreateHostRandTensor!";
    }

    return tensor;
}

void OpTest::GenerateRandomTensors(const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs,
                                   AsdOps::SVector<AsdOps::Tensor> &inTensors)
{
    if (inTensorDescs.size() != inTensors.size()) {
        ASD_LOG(ERROR) << "TensorDescs Num not equal Tensors Num!";
        return;
    }
    for (size_t i = 0; i < inTensorDescs.size(); i++) {
        inTensors.at(i) = CreateHostRandTensor(inTensorDescs.at(i));
    }
}

void OpTest::Golden(OpTestGolden golden) { golden_ = golden; }

void OpTest::FloatRand(float min, float max)
{
    randFloatMin_ = min;
    randFloatMax_ = max;
    ASD_LOG(INFO) << "randFloatMin:" << randFloatMin_ << ", randFloatMax:" << randFloatMax_;
}

void OpTest::Int8Rand(int8_t min, int8_t max)
{
    randInt8Min_ = min;
    randInt8Max_ = max;
    ASD_LOG(INFO) << "randIntMin:" << randInt8Min_ << ", randIntMax:" << randInt8Max_;
}

void OpTest::IntRand(int32_t min, int32_t max)
{
    randIntMin_ = min;
    randIntMax_ = max;
    ASD_LOG(INFO) << "randIntMin:" << randIntMin_ << ", randIntMax:" << randIntMax_;
}

void OpTest::LongRand(int64_t min, int64_t max)
{
    randLongMin_ = min;
    randLongMax_ = max;
    ASD_LOG(INFO) << "randIntMin:" << randLongMin_ << ", randIntMax:" << randLongMax_;
}
} // namespace AclTransformer