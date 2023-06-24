
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
#include "example_util.h"
#include <iostream>
#include <sys/stat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/filesystem/filesystem.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/plan_builder.h"
#include "acltransformer/config.h"
#include "acltransformer/utils/tensor_util.h"

void *ExampleUtil::GetCurrentStream()
{
    int32_t devId = 0;
    AsdRtDeviceGetCurrent(&devId);
    void *stream = c10_npu::getCurrentNPUStream(devId).stream();
    ASD_LOG_IF(stream == nullptr, ERROR) << "get current stream fail";
    return stream;
}

int64_t ExampleUtil::GetTensorNpuFormat(const at::Tensor &tensor)
{
#ifdef TORCH_GET_TENSOR_NPU_FORMAT_OLD
    return at_npu::native::CalcuOpUtil::get_tensor_npu_format(tensor);
#else
    return at_npu::native::CalcuOpUtil::GetTensorNpuFormat(tensor);
#endif
}

at::Tensor ExampleUtil::NpuFormatCast(const at::Tensor &tensor)
{
    return at_npu::native::NPUNativeFunctions::npu_format_cast(tensor, GetTensorNpuFormat(tensor));
}

void ExampleUtil::ExecuteRunner(AclTransformer::Runner *runner, std::vector<at::Tensor> atInTensors,
                                std::vector<at::Tensor> atOutTensors)
{
    AclTransformer::Handle handle;
    handle.stream = GetCurrentStream();

    AclTransformer::VariantPack variantPack;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2AsdTensor(atInTensors.at(i)));
    }
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2AsdTensor(atOutTensors.at(i)));
    }

    runner->Setup(variantPack);
    variantPack.workspaceSize = runner->GetWorkspaceSize();

    if (variantPack.workspaceSize > 0) {
        int st = AsdRtMemMallocDevice((void **)&variantPack.workspace, variantPack.workspaceSize, ASDRT_MEM_DEFAULT);
        if (st != ASDRT_SUCCESS) {
            ASD_LOG(ERROR) << "malloc device memory fail";
            return;
        }
    }

    runner->Execute(handle, variantPack);
    ASD_LOG(INFO) << "AsdRtStreamSynchronize stream:" << handle.stream;
    AsdRtStreamSynchronize(handle.stream);

    if (variantPack.workspace != nullptr) {
        AsdRtMemFreeDevice(variantPack.workspace);
        ASD_LOG(INFO) << "AsdRtMemFreeDevice free:" << variantPack.workspace
                      << ", variantPack.workspaceSize:" << variantPack.workspaceSize;
        variantPack.workspace = nullptr;
        variantPack.workspaceSize = 0;
    }
}

void ExampleUtil::ExecuteOperation(AclTransformer::Operation *operation, std::vector<at::Tensor *> atInTensors,
                                   std::vector<at::Tensor *> atOutTensors)
{
    AclTransformer::Handle handle = {GetCurrentStream()};
    AclTransformer::VariantPack variantPack;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2AsdTensor(*atInTensors.at(i)));
    }
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2AsdTensor(*atOutTensors.at(i)));
    }

    static int64_t opId = 0;
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
        std::string dirPath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(opId++) + "_" +
                              operation->GetName() + "_brefore";
        AclTransformer::TensorUtil::SaveVariantPack(handle, variantPack, dirPath);
        ASD_LOG(INFO) << operation->GetName() << " SaveVariantPack " << dirPath;
    }

    AsdOps::Status st = operation->Setup(variantPack);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << operation->GetName() << " Setup fail, not call execute";
        return;
    }

    variantPack.workspaceSize = operation->GetWorkspaceSize();
    ASD_LOG(ERROR) << operation->GetName() << " GetWorkspaceSize:" << variantPack.workspaceSize;
    if (variantPack.workspaceSize > 0) {
        int st = AsdRtMemMallocDevice((void **)&variantPack.workspace, variantPack.workspaceSize, ASDRT_MEM_DEFAULT);
        if (st != ASDRT_SUCCESS) {
            ASD_LOG(ERROR) << operation->GetName() << " AsdRtMemMallocDevice fail";
            return;
        }
    }

    st = operation->Execute(handle, variantPack);
    ASD_LOG_IF(!st.Ok(), ERROR) << operation->GetName() << " execute fail, error:" << st.Message();

    if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
        std::string dirPath =
            AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(opId++) + "_" + operation->GetName();
        AclTransformer::TensorUtil::SaveVariantPack(handle, variantPack, dirPath);
        ASD_LOG(INFO) << operation->GetName() << " SaveVariantPack " << dirPath;
    }

    if (variantPack.workspace != nullptr) {
        AsdRtMemFreeDevice(variantPack.workspace);
        ASD_LOG(INFO) << operation->GetName() << " AsdRtMemFreeDevice free:" << variantPack.workspace;
        variantPack.workspace = nullptr;
        variantPack.workspaceSize = 0;
    }
}

void ExampleUtil::BuildVariantPack(const std::vector<torch::Tensor> &inTensors,
                                   const std::vector<torch::Tensor> &outTensors,
                                   AclTransformer::VariantPack &variantPack)
{
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2AsdTensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2AsdTensor(outTensors.at(i)));
    }
}

AsdOps::Tensor ExampleUtil::AtTensor2AsdTensor(const at::Tensor &atTensor)
{
    static std::map<at::ScalarType, AsdOps::TensorDType> dtypeMap = {
        {at::ScalarType::Bool, AsdOps::TENSOR_DTYPE_BOOL},   {at::ScalarType::Byte, AsdOps::TENSOR_DTYPE_UINT8},
        {at::ScalarType::Char, AsdOps::TENSOR_DTYPE_UINT8},  {at::ScalarType::Half, AsdOps::TENSOR_DTYPE_FLOAT16},
        {at::ScalarType::Float, AsdOps::TENSOR_DTYPE_FLOAT}, {at::ScalarType::Int, AsdOps::TENSOR_DTYPE_INT32},
        {at::ScalarType::Long, AsdOps::TENSOR_DTYPE_INT64},
    };

    ASD_LOG_IF(!atTensor.is_contiguous(), ERROR) << "atTensor is not contiguous";
    AsdOps::Tensor asdTensor;
    asdTensor.desc.format = static_cast<AsdOps::TensorFormat>(GetTensorNpuFormat(atTensor));
    asdTensor.data = atTensor.data_ptr();

    asdTensor.desc.dims.resize(atTensor.sizes().size());
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        asdTensor.desc.dims[i] = atTensor.sizes()[i];
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    if (it != dtypeMap.end()) {
        asdTensor.desc.dtype = it->second;
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }

    asdTensor.dataSize = AclTransformer::TensorUtil::CalcTensorDataSize(asdTensor);

    return asdTensor;
}

at::Tensor ExampleUtil::CreateAtTensorFromAsdOpsTensorDesc(const AsdOps::TensorDesc &tensorDesc)
{
    at::TensorOptions options = at::TensorOptions();
    if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_FLOAT) {
        options = options.dtype(at::kFloat);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_FLOAT16) {
        options = options.dtype(at::kHalf);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_BOOL) {
        options = options.dtype(at::kBool);
    } else if (tensorDesc.dtype == AsdOps::TENSOR_DTYPE_INT64) {
        options = options.dtype(at::kLong);
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << tensorDesc.dtype;
    }

#ifdef TORCH_18
    options = options.layout(torch::kStrided).requires_grad(false).device(at::DeviceType::XLA);
#else
    options = options.layout(torch::kStrided).requires_grad(false).device(at::kPrivateUse1);
#endif

    ASD_LOG(INFO) << "ApplyTensorWithFormat stat, format:" << tensorDesc.format;
    at::Tensor newTensor = at_npu::native::OpPreparation::ApplyTensorWithFormat(
        at::IntArrayRef(tensorDesc.dims.data(), tensorDesc.dims.size()), options, tensorDesc.format);
    ASD_LOG(INFO) << "ApplyTensorWithFormat end, newTensor.format:" << GetTensorNpuFormat(newTensor)
                  << ", is_contiguous:" << newTensor.is_contiguous();
    if (GetTensorNpuFormat(newTensor) != tensorDesc.format) {
        ASD_LOG(WARN) << "ApplyTensorWithFormat newTensor.format:" << GetTensorNpuFormat(newTensor)
                      << " != " << tensorDesc.format;
    }
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }

    ASD_LOG(INFO) << "ApplyTensorWithFormat success, newTensor.options:" << newTensor.options()
                  << ", format:" << GetTensorNpuFormat(newTensor) << ", is_contiguous:" << newTensor.is_contiguous();

    return newTensor;
}

void ExampleUtil::SaveTensor(const at::Tensor &tensor, const std::string &filePath)
{
    std::string dirPath = AsdOps::FileSystem::DirName(filePath);
    if (!AsdOps::FileSystem::Exists(dirPath)) {
        ASD_LOG(INFO) << "create dir:" << dirPath;
        AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    torch::save(tensor.to(at::Device(at::kCPU)), filePath);
}