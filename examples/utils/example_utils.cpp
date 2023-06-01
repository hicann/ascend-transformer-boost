
/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "example_utils.h"
#include <asdops/utils/rt/rt.h>
#include <iostream>
#include <asdops/utils/log/log.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "acltransformer/plan_builder.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"

static std::map<at::ScalarType, AsdOps::TensorDType> DTYPE_MAP = {
    {at::ScalarType::Byte, AsdOps::TENSOR_DTYPE_UINT8},   {at::ScalarType::Char, AsdOps::TENSOR_DTYPE_UINT8},
    {at::ScalarType::Half, AsdOps::TENSOR_DTYPE_FLOAT16}, {at::ScalarType::Float, AsdOps::TENSOR_DTYPE_FLOAT},
    {at::ScalarType::Int, AsdOps::TENSOR_DTYPE_INT32},    {at::ScalarType::Long, AsdOps::TENSOR_DTYPE_INT64},
};

void *GetCurrentStream()
{
    int32_t devId = 0;
    AsdRtDeviceGetCurrent(&devId);
    void *stream = c10_npu::getCurrentNPUStream(devId).stream();
    ASD_LOG_IF(stream == nullptr, ERROR) << "get current stream fail";
    return stream;
}

uint64_t CalcTensorDataSize(const AsdOps::Tensor &tensor)
{
    uint64_t dataItemSize = 0;
    switch (tensor.desc.dtype) {
    case AsdOps::TENSOR_DTYPE_FLOAT: dataItemSize = sizeof(float); break;
    case AsdOps::TENSOR_DTYPE_FLOAT16: dataItemSize = 2; break;
    default: ASD_LOG(ERROR) << "not support dtype:" << tensor.desc.dtype;
    }

    return dataItemSize * tensor.Numel();
}

AsdOps::Tensor AtTensor2AsdTensor(const at::Tensor &atTensor)
{
    AsdOps::Tensor asdTensor;
    asdTensor.desc.format = AsdOps::TENSOR_FORMAT_ND;
    asdTensor.data = atTensor.storage().data_ptr().get();

    asdTensor.desc.dims.resize(atTensor.sizes().size());
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        asdTensor.desc.dims[i] = atTensor.sizes()[i];
    }

    auto it = DTYPE_MAP.find(atTensor.scalar_type());
    if (it != DTYPE_MAP.end()) {
        asdTensor.desc.dtype = it->second;
    } else {
        ASD_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }

    asdTensor.dataSize = CalcTensorDataSize(asdTensor);

    return asdTensor;
}

void ExecuteRunner(AclTransformer::Runner *runner, std::vector<at::Tensor> atInTensors,
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

void ExecuteOperation(AclTransformer::Operation *operation, std::vector<at::Tensor> atInTensors,
                      std::vector<at::Tensor> atOutTensors)
{
    AclTransformer::Handle handle = {GetCurrentStream()};
    AclTransformer::VariantPack variantPack;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2AsdTensor(atInTensors.at(i)));
    }
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2AsdTensor(atOutTensors.at(i)));
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

    static int64_t opId = 0;
    if (AclTransformer::Config::IsSaveTensor()) {
        std::string dirPath = "savetensor/" + std::to_string(opId++) + "_" + operation->GetName();
        SaveVariantPack(variantPack, dirPath);
        ASD_LOG(INFO) << operation->GetName() << " SaveVariantPack " << dirPath;
    }

    if (variantPack.workspace != nullptr) {
        AsdRtMemFreeDevice(variantPack.workspace);
        ASD_LOG(INFO) << operation->GetName() << " AsdRtMemFreeDevice free:" << variantPack.workspace;
        variantPack.workspace = nullptr;
        variantPack.workspaceSize = 0;
    }
}

void ExecuteOperationGraph(AclTransformer::OperationGraph &opGraph, AclTransformer::VariantPack &variantPack)
{
    AclTransformer::Handle handle = {GetCurrentStream()};

    AclTransformer::PlanBuilder planBuilder;
    AclTransformer::Plan plan;
    AsdOps::Status st = planBuilder.Build(variantPack, opGraph, plan);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << opGraph.name << " PlanBuilder build plan fail, error:" << st.Message();
        return;
    }

    st = plan.Setup(handle, variantPack);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << opGraph.name << " Plan Setup fail error:" << st.Message();
        return;
    }

    variantPack.workspaceSize = plan.GetWorkspaceSize();
    ASD_LOG(INFO) << opGraph.name << " Plan GetWorkspaceSize:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        ASD_LOG(INFO) << opGraph.name
                      << " AsdRtMemMallocDevice variantPack.workspaceSize:" << variantPack.workspaceSize;
        int st = AsdRtMemMallocDevice((void **)&variantPack.workspace, variantPack.workspaceSize, ASDRT_MEM_DEFAULT);
        if (st != ASDRT_SUCCESS) {
            ASD_LOG(ERROR) << opGraph.name << " AsdRtMemMallocDevice fail";
            return;
        }
    }

    st = plan.Execute(handle, variantPack);
    ASD_LOG_IF(!st.Ok(), ERROR) << opGraph.name << " Plan Execute fail, error:" << st.Message();

    if (variantPack.workspace != nullptr) {
        AsdRtMemFreeDevice(variantPack.workspace);
        ASD_LOG(INFO) << opGraph.name << " AsdRtMemFreeDevice free:" << variantPack.workspace;
        variantPack.workspace = nullptr;
        variantPack.workspaceSize = 0;
    }
}

std::string TensorToString(const AsdOps::Tensor &tensor)
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
                // half_float::half *tensorData = static_cast<half_float::half *>(tensor.data);
                // ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == AsdOps::TENSOR_DTYPE_FLOAT) {
                float *tensorData = static_cast<float *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == AsdOps::TENSOR_DTYPE_INT32) {
                int32_t *tensorData = static_cast<int32_t *>(tensor.data);
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

void BuildVariantPack(const std::vector<torch::Tensor> &inTensors, const std::vector<torch::Tensor> &outTensors,
                      AclTransformer::VariantPack &variantPack)
{
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2AsdTensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2AsdTensor(outTensors.at(i)));
    }
}