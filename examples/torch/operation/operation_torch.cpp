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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#pragma GCC diagnostic pop
#include <torch_npu/csrc/core/npu/register/OptionsManager.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/ops.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "acltransformer/statistic.h"
#include "examples/utils/example_util.h"
#include "examples/workspace/workspace.h"
#include "operation_creator.h"

OperationTorch::OperationTorch(std::string opName) : opName_(opName), name_(opName)
{
    ASD_LOG(INFO) << "OperationTorch::OperationTorch, TASK_QUEUE_ENABLE:"
                  << c10_npu::option::OptionsManager().CheckQueueEnable() << ", opName:" << opName;
    std::vector<AsdOps::Operation *> ops;
    AsdOps::Ops::Instance().GetAllOperations(ops);
}

OperationTorch::~OperationTorch() {}

void OperationTorch::SetName(std::string name) { name_ = name; }

void OperationTorch::SetParam(std::string param)
{
    ASD_LOG(INFO) << name_ << " set param start, param:" << param;
    param_ = param;

    AclTransformer::Operation *operation = CreateOperation(opName_, param_);
    if (operation == nullptr) {
        ASD_LOG(FATAL) << name_ << " create operation fail, opName:" << opName_ << ", param:" << param_;
        return;
    }

    operation_.reset(operation);

    AsdOps::Status st = operation_->BuildPlan(&planv2_);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " build plan fail, error:" << st.Message();
        return;
    }

    ASD_LOG(INFO) << name_ << " set param end";
}

std::vector<torch::Tensor> OperationTorch::Execute(std::vector<torch::Tensor> atInTensors)
{
    ASD_LOG(INFO) << name_ << " execute start";
    if (!operation_) {
        ASD_LOG(FATAL) << name_ << " execute fail, operation is null";
    }
    ExampleUtil::ContiguousAtTensor(atInTensors);

    std::vector<torch::Tensor> atOutTensors;
    CreateAtOutTensors(atInTensors, atOutTensors);
    ExampleUtil::ContiguousAtTensor(atOutTensors);

    ExecuteOutImpl(atInTensors, atOutTensors);
    return atOutTensors;
}

void OperationTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors)
{
    ASD_LOG(INFO) << name_ << " execute out start";
    if (!operation_) {
        ASD_LOG(FATAL) << name_ << " execute out fail, operation is null";
    }

    ExampleUtil::ContiguousAtTensor(atInTensors);
    ExampleUtil::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors);
}

void OperationTorch::ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors)
{
    ASD_LOG(INFO) << name_ << " execute impl execCount:" << executeCount_;
    AsdOps::Timer timer;

    AclTransformer::Handle handle = {ExampleUtil::GetCurrentStream()};

    AclTransformer::VariantPack variantPack;
    BuildVariantPack(atInTensors, atOutTensors, variantPack);

    AsdOps::Timer timer1;
    AsdOps::Status st = planv2_.Setup(handle, variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime += timer1.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " setup plan fail, not call execute";
        return;
    }

    variantPack.workspaceSize = planv2_.GetWorkspaceSize();
    ASD_LOG(INFO) << name_ << " get plan workspace size:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        AsdOps::GetSingleton<AclTransformer::Workspace>().SetWorkspace(variantPack.workspaceSize);
        variantPack.workspace = AsdOps::GetSingleton<AclTransformer::Workspace>().GetWorkspace();
    }

    AsdOps::Timer timer2;
    st = planv2_.Execute(handle, variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planExecuteTime += timer2.ElapsedMicroSecond();
    ASD_LOG_IF(!st.Ok(), ERROR) << name_ << " execute plan fail, error:" << st.Message();

    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + opName_ + "/outtensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atOutTensors.at(i), filePath);
            ASD_LOG(INFO) << name_ << " save tensor:" << filePath;
        }
    }

    AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime += timer.ElapsedMicroSecond();
    ASD_LOG(FATAL) << name_ << " executeCount:" << executeCount_ << ", statistic:["
                   << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString() << "]";
    AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();

    executeCount_++;
}

void OperationTorch::CreateAtOutTensors(const std::vector<torch::Tensor> &atInTensors,
                                        std::vector<torch::Tensor> &atOutTensors)
{
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;

    AsdOps::SVector<AsdOps::Tensor> inTensors;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        auto &atInTensor = atInTensors.at(i);
        AsdOps::Tensor inTensor = ExampleUtil::AtTensor2AsdTensor(atInTensor);
        inTensors.push_back(inTensor);
        ASD_LOG(INFO) << name_ << " infer shape inTensors[" << i
                      << "]:" << AclTransformer::TensorUtil::AsdOpsTensorToString(inTensors.at(i));
    }
    operation_->InferShape(inTensors, outTensorDescs);

    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ASD_LOG(INFO) << name_ << " infer shape outTensorDescs[" << i
                      << "]:" << AclTransformer::TensorUtil::AsdOpsTensorDescToString(outTensorDescs.at(i));
        at::Tensor newTensor = ExampleUtil::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(i));
        atOutTensors.at(i) = newTensor;
    }
}

void OperationTorch::BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                                      AclTransformer::VariantPack &variantPack)
{
    variantPack.inTensors.resize(atInTensors.size());
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << name_ << " execute start, atInTensors[" << i << "].options:" << atInTensors.at(i).options()
                      << ", data:" << atInTensors.at(i).data_ptr()
                      << ", storage_offset:" << atInTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atInTensors.at(i));
        variantPack.inTensors.at(i) = ExampleUtil::AtTensor2AsdTensor(atInTensors.at(i));
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
            variantPack.inTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
            variantPack.inTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + opName_ + "/intensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atInTensors.at(i), filePath);
            ASD_LOG(INFO) << operation_->GetName() << " save tensor:" << filePath;
        }
    }

    variantPack.outTensors.resize(atOutTensors.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ASD_LOG(INFO) << name_ << " execute start, atOutTensors[" << i << "].options:" << atOutTensors.at(i).options()
                      << ", data:" << atOutTensors.at(i).data_ptr()
                      << ", storage_offset:" << atOutTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atOutTensors.at(i));
        variantPack.outTensors.at(i) = ExampleUtil::AtTensor2AsdTensor(atOutTensors.at(i));
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND() &&
            variantPack.outTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
            variantPack.outTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + opName_ + "/outtensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atOutTensors.at(i), filePath);
            ASD_LOG(INFO) << operation_->GetName() << " save tensor:" << filePath;
        }
    }
}

TORCH_LIBRARY(OperationTorch, m)
{
    m.class_<OperationTorch>("OperationTorch")
        .def(torch::init<std::string>())
        .def("set_name", &OperationTorch::SetName)
        .def("set_param", &OperationTorch::SetParam)
        .def("execute", &OperationTorch::Execute)
        .def("execute_out", &OperationTorch::ExecuteOut);
}