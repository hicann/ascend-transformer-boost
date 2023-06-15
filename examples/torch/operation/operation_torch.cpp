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
#include "operation_torch.h"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/core/npu/register/OptionsManager.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "examples/utils/example_util.h"
#include "operation_creator.h"

OperationTorch::OperationTorch(std::string opName) : opName_(opName)
{
    ASD_LOG(INFO) << "OperationTorch::OperationTorch, TASK_QUEUE_ENABLE:"
                  << c10_npu::option::OptionsManager().CheckQueueEnable();
}

OperationTorch::~OperationTorch() {}

void OperationTorch::SetParam(std::string param) { param_ = param; }

std::vector<torch::Tensor> OperationTorch::Execute(std::vector<torch::Tensor> atInTensors)
{
    ASD_LOG(INFO) << "OperationTorch::Execute";
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
    ASD_LOG(INFO) << "OperationTorch::ExecuteOperation";
    static int64_t execCount = 0;
    AclTransformer::Handle handle = {ExampleUtil::GetCurrentStream()};
    AclTransformer::VariantPack variantPack;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << "inTensors[" << i << "].options:" << atInTensors.at(i).options()
                      << ", data:" << atInTensors.at(i).data_ptr()
                      << ", storage_offset:" << atInTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atInTensors.at(i));
        variantPack.inTensors.push_back(ExampleUtil::AtTensor2AsdTensor(atInTensors.at(i)));
        if (AclTransformer::Config::IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(execCount) + "_" +
                                   opName_ + "/intensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atInTensors.at(i), filePath);
            ASD_LOG(INFO) << operation->GetName() << " save tensor:" << filePath;
        }
    }

    CreateAtOutTensors(operation, variantPack.inTensors, atOutTensors);

    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ASD_LOG(INFO) << "atOutTensors[" << i << "].options:" << atOutTensors.at(i).options()
                      << ", data:" << atOutTensors.at(i).data_ptr()
                      << ", storage_offset:" << atOutTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atOutTensors.at(i));
        variantPack.outTensors.push_back(ExampleUtil::AtTensor2AsdTensor(atOutTensors.at(i)));
    }

    AsdOps::Status st = operation->Setup(variantPack);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << operation->GetName() << " Setup fail, not call execute";
        return;
    }

    variantPack.workspaceSize = operation->GetWorkspaceSize();
    ASD_LOG(INFO) << operation->GetName() << " GetWorkspaceSize:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        int st = AsdRtMemMallocDevice((void **)&variantPack.workspace, variantPack.workspaceSize, ASDRT_MEM_DEFAULT);
        if (st != ASDRT_SUCCESS) {
            ASD_LOG(ERROR) << operation->GetName() << " AsdRtMemMallocDevice fail";
            return;
        }
    }

    st = operation->Execute(handle, variantPack);
    ASD_LOG_IF(!st.Ok(), ERROR) << operation->GetName() << " execute fail, error:" << st.Message();

    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        if (AclTransformer::Config::IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(execCount) + "_" +
                                   opName_ + "/outtensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atOutTensors.at(i), filePath);
            ASD_LOG(INFO) << operation->GetName() << " save tensor:" << filePath;
        }
    }

    if (variantPack.workspace != nullptr) {
        AsdRtMemFreeDevice(variantPack.workspace);
        ASD_LOG(INFO) << operation->GetName() << " AsdRtMemFreeDevice free:" << variantPack.workspace;
        variantPack.workspace = nullptr;
        variantPack.workspaceSize = 0;
    }
    execCount++;
}

void OperationTorch::CreateAtOutTensors(AclTransformer::Operation *operation,
                                        const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                        std::vector<torch::Tensor> &atOutTensors)
{
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    operation->InferShape(inTensors, outTensorDescs);

    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        at::Tensor newTensor = ExampleUtil::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(i));
        atOutTensors.at(i) = newTensor;
    }
}

TORCH_LIBRARY(OperationTorch, m)
{
    m.class_<OperationTorch>("OperationTorch")
        .def(torch::init<std::string>())
        .def("set_param", &OperationTorch::SetParam)
        .def("execute", &OperationTorch::Execute);
}