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
#include "acltransformer/config.h"
#include "examples/utils/example_utils.h"
#include "operation_creator.h"

OperationTorch::OperationTorch() { ASD_LOG(INFO) << "OperationTorch::OperationTorch"; }

OperationTorch::~OperationTorch() {}

void OperationTorch::Test() { ASD_LOG(INFO) << "OperationTorch::Test called"; }

void OperationTorch::Execute(std::string opName, std::string param, std::vector<torch::Tensor> atInTensors,
                             std::vector<torch::Tensor> atOutTensors)
{
    AclTransformer::Operation *operation = CreateOperation(opName, param);
    if (operation == nullptr) {
        ASD_LOG(ERROR) << "create operation fail, json:" << param;
        return;
    }

    ExecuteOperation(operation, atInTensors, atOutTensors);
    delete operation;
}

void OperationTorch::ExecuteOperation(AclTransformer::Operation *operation, std::vector<torch::Tensor> atInTensors,
                                      std::vector<torch::Tensor> atOutTensors)
{
    AclTransformer::Handle handle = {GetCurrentStream()};
    AclTransformer::VariantPack variantPack;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        atInTensors.at(i) = atInTensors.at(i).contiguous();
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

TORCH_LIBRARY(OperationTorch, m)
{
    m.class_<OperationTorch>("OperationTorch")
        .def(torch::init<>())
        .def("test", &OperationTorch::Test)
        .def("execute", &OperationTorch::Execute);
}