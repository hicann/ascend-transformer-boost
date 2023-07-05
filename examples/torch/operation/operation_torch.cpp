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
#include <asdops/ops.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "acltransformer/statistic.h"
#include "acltransformer/context/work_context.h"
#include "examples/utils/example_util.h"
#include "operation_creator.h"

uint64_t OperationTorch::totalExecuteCount_ = 0;

OperationTorch::OperationTorch(std::string opName) : name_(opName), opName_(opName)
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
    ASD_LOG(INFO) << "OperationTorch::SetParam start, param:" << param;
    param_ = param;

    AclTransformer::Operation *operation = CreateOperation(opName_, param_);
    if (operation == nullptr) {
        ASD_LOG(FATAL) << "create operation fail, opName:" << opName_ << ", param:" << param_;
        return;
    }

    operation_.reset(operation);

    AsdOps::Status st = operation_->BuildPlan(&plan_);
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " build plan fail, error:" << st.Message();
        return;
    }

    ASD_LOG(INFO) << "OperationTorch::SetParam end";
}

std::vector<torch::Tensor> OperationTorch::Execute(std::vector<torch::Tensor> atInTensors)
{
    timer_.Reset();
    ASD_LOG(INFO) << name_ << " Execute start";
    if (!operation_) {
        ASD_LOG(FATAL) << name_ << " Execute fail, operation is null";
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
    timer_.Reset();
    ASD_LOG(INFO) << name_ << " ExecuteOut start";
    if (!operation_) {
        ASD_LOG(FATAL) << name_ << " ExecuteOut fail, operation is null";
    }

    ExampleUtil::ContiguousAtTensor(atInTensors);
    ExampleUtil::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors);
}

void OperationTorch::ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors)
{
    AclTransformer::Handle handle = {ExampleUtil::GetCurrentStream()};

    AclTransformer::VariantPack variantPack;
    BuildVariantPack(atInTensors, atOutTensors, variantPack);

    AsdOps::Timer t1;
    AsdOps::Status st = plan_.Setup(variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime += t1.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << name_ << " Setup fail, not call execute";
        return;
    }

    if (!CopyTiling(handle, variantPack)) {
        return;
    }

    variantPack.workspaceBufferSize = plan_.GetWorkspaceBufferSize();
    ASD_LOG(INFO) << name_ << " GetWorkspaceBufferSize:" << variantPack.workspaceBufferSize;
    if (variantPack.workspaceBufferSize > 0) {
        variantPack.workspaceBuffer =
            AsdOps::GetSingleton<AclTransformer::WorkContext>().GetWorkspaceBuffer(variantPack.workspaceBufferSize);
    }

    variantPack.intermediateBufferSize = plan_.GetIntermediateBufferSize();
    ASD_LOG(INFO) << name_ << " GetIntermediateBufferSize:" << variantPack.intermediateBufferSize;
    if (variantPack.intermediateBufferSize) {
        variantPack.intermediateBuffer = AsdOps::GetSingleton<AclTransformer::WorkContext>().GetIntermediateBuffer(
            variantPack.intermediateBufferSize);
    }

    AsdOps::Timer t2;
    st = plan_.Execute(handle, variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planExecuteTime += t2.ElapsedMicroSecond();
    ASD_LOG_IF(!st.Ok(), ERROR) << name_ << " execute fail, error:" << st.Message();

    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + name_ + "/outtensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atOutTensors.at(i), filePath);
            ASD_LOG(INFO) << name_ << " save tensor:" << filePath;
        }
    }

    AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime += timer_.ElapsedMicroSecond();
    ASD_LOG(FATAL) << name_ << " totalExecuteCount:" << totalExecuteCount_ << ", executeCount:" << executeCount_
                   << ", statistic:[" << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString() << "]";
    AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();

    executeCount_++;
    totalExecuteCount_++;
}

bool OperationTorch::CopyTiling(AclTransformer::Handle handle, AclTransformer::VariantPack &variantPack)
{
    variantPack.tilingBufferSize = plan_.GetTilingBufferSize();
    ASD_LOG(INFO) << name_ << " GetTilingBufferSize:" << variantPack.tilingBufferSize;

    if (variantPack.tilingBufferSize == 0) {
        return true;
    }

    void *hostTilingBuffer =
        AsdOps::GetSingleton<AclTransformer::WorkContext>().GetHostTilingBuffer(variantPack.tilingBufferSize);
    if (hostTilingBuffer == nullptr) {
        ASD_LOG(FATAL) << name_ << " GetHostTilingBuffer null buffer";
        return false;
    }

    AsdOps::Timer t1;
    plan_.FillHostTilingBufferSize(hostTilingBuffer, variantPack.tilingBufferSize);
    AsdOps::GetSingleton<AclTransformer::Statistic>().tilingFillTime += t1.ElapsedMicroSecond();

    variantPack.tilingBuffer =
        AsdOps::GetSingleton<AclTransformer::WorkContext>().GetTilingBuffer(variantPack.tilingBufferSize);
    if (variantPack.tilingBuffer == nullptr) {
        ASD_LOG(FATAL) << name_ << " GetTilingBuffer null buffer";
        return false;
    }

    AsdOps::Timer timer;
    int ret = AsdRtMemCopyAsync(variantPack.tilingBuffer, variantPack.tilingBufferSize, hostTilingBuffer,
                                variantPack.tilingBufferSize, ASDRT_MEMCOPY_HOST_TO_DEVICE, handle.stream);
    AsdOps::GetSingleton<AclTransformer::Statistic>().tillingCopyTime += timer.ElapsedMicroSecond();
    if (ret != 0) {
        ASD_LOG(ERROR) << name_ << " copy host tiling to device fail, ret:" << ret;
        return false;
    }

    return true;
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
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << "inTensors[" << i << "].options:" << atInTensors.at(i).options()
                      << ", data:" << atInTensors.at(i).data_ptr()
                      << ", storage_offset:" << atInTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atInTensors.at(i));
        atInTensors.at(i) = ExampleUtil::NpuFormatCast(atInTensors.at(i));
        variantPack.inTensors.push_back(ExampleUtil::AtTensor2AsdTensor(atInTensors.at(i)));
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + name_ + "/intensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atInTensors.at(i), filePath);
            ASD_LOG(INFO) << name_ << " save tensor:" << filePath;
        }
    }

    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ASD_LOG(INFO) << "atOutTensors[" << i << "].options:" << atOutTensors.at(i).options()
                      << ", data:" << atOutTensors.at(i).data_ptr()
                      << ", storage_offset:" << atOutTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atOutTensors.at(i));
        variantPack.outTensors.push_back(ExampleUtil::AtTensor2AsdTensor(atOutTensors.at(i)));
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + name_ + "/outtensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atOutTensors.at(i), filePath);
            ASD_LOG(INFO) << name_ << " save tensor:" << filePath;
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