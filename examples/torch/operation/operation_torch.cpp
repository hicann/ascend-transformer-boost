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
#include "operation_torch.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/utils/tensor_cache.h"
#include "acltransformer/config.h"
#include "examples/utils/example_utils.h"
#include "operation_creator.h"

OperationTorch::OperationTorch(std::string opName) : opName_(opName)
{
    ASD_LOG(INFO) << "OperationTorch::OperationTorch";
}

OperationTorch::~OperationTorch() {}

void OperationTorch::SetParam(std::string param) { param_ = param; }

std::vector<torch::Tensor> OperationTorch::Execute(std::vector<torch::Tensor> atInTensors)
{
    for (auto &inTensor : atInTensors) {
        inTensor = inTensor.contiguous();
    }

    std::vector<torch::Tensor> atOutTensors;

    AclTransformer::Operation *operation = CreateOperation(opName_, param_);
    if (operation == nullptr) {
        ASD_LOG(ERROR) << "create operation fail, json:" << param_;
        return atOutTensors;
    }

    ExecuteOperation(operation, atInTensors, atOutTensors);

    delete operation;
    return atOutTensors;
}

void OperationTorch::ExecuteOperation(AclTransformer::Operation *operation, std::vector<torch::Tensor> &atInTensors,
                                      std::vector<torch::Tensor> &atOutTensors)
{
    AclTransformer::Handle handle = {GetCurrentStream()};
    AclTransformer::VariantPack variantPack;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        variantPack.inTensors.push_back(AclTransformer::AtTensor2AsdTensor(atInTensors.at(i)));
        AsdOps::GetSingleton<AclTransformer::TensorCache>().AddTensor(atInTensors.at(i).data_ptr(), &atInTensors.at(i));
    }

    CreateAtOutTensors(operation, variantPack.inTensors, atOutTensors);
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        variantPack.outTensors.push_back(AclTransformer::AtTensor2AsdTensor(atOutTensors.at(i)));
        AsdOps::GetSingleton<AclTransformer::TensorCache>().AddTensor(atOutTensors.at(i).data_ptr(),
                                                                      &atOutTensors.at(i));
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
        SaveVariantPack(handle, variantPack, dirPath);
        ASD_LOG(INFO) << operation->GetName() << " SaveVariantPack " << dirPath;
    }

    if (variantPack.workspace != nullptr) {
        AsdRtMemFreeDevice(variantPack.workspace);
        ASD_LOG(INFO) << operation->GetName() << " AsdRtMemFreeDevice free:" << variantPack.workspace;
        variantPack.workspace = nullptr;
        variantPack.workspaceSize = 0;
    }

    for (size_t i = 0; i < atInTensors.size(); ++i) {
        AsdOps::GetSingleton<AclTransformer::TensorCache>().DeleteTensor(atInTensors.at(i).data_ptr());
    }
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        AsdOps::GetSingleton<AclTransformer::TensorCache>().DeleteTensor(atOutTensors.at(i).data_ptr());
    }
}

void OperationTorch::CreateAtOutTensors(AclTransformer::Operation *operation,
                                        const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                        std::vector<torch::Tensor> &atOutTensors)
{
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    operation->InferShape(inTensors, outTensorDescs);

    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        at::Tensor newTensor = AclTransformer::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(i));
        atOutTensors.at(i) = newTensor.contiguous();
    }
}

TORCH_LIBRARY(OperationTorch, m)
{
    m.class_<OperationTorch>("OperationTorch")
        .def(torch::init<std::string>())
        .def("set_param", &OperationTorch::SetParam)
        .def("execute", &OperationTorch::Execute);
}