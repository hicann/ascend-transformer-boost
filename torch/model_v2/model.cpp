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
#include "model.h"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#pragma GCC diagnostic pop
#include <torch_npu/csrc/core/npu/register/OptionsManager.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/ops.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "acltransformer/statistic.h"
#include "torch/utils/utils.h"
#include "acltransformer/context/context.h"

namespace AclTransformer {
std::string Model::Graph::ToString() const
{
    std::stringstream ss;
    for (size_t i = 0; i < weightTensors.size(); ++i) {
        ss << "weightTensors[" << i << "]:" << &weightTensors.at(i) << " "
           << TensorUtil::AsdOpsTensorToString(weightTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " " << TensorUtil::AsdOpsTensorToString(inTensors.at(i))
           << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " "
           << TensorUtil::AsdOpsTensorToString(outTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]:" << &internalTensors.at(i) << " "
           << TensorUtil::AsdOpsTensorToString(internalTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto &node = nodes.at(i);
        ss << "node[" << i << "] opeation:" << node.operation.get()
           << ", operationName:" << (node.operation ? node.operation->GetName() : "null") << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt << " " << TensorUtil::AsdOpsTensorToString(*tensorIt)
               << std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt << " " << TensorUtil::AsdOpsTensorToString(*tensorIt)
               << std::endl;
        }
    }
    return ss.str();
}

void Model::Graph::Init()
{
    for (size_t i = 0; i < nodes.size(); i++) {
        auto &node = nodes.at(i);
        node.plan = std::make_shared<Plan>();
        node.operation->BuildPlan(node.plan.get());
        node.variantPack.inTensors.resize(node.inTensors.size());
        node.variantPack.outTensors.resize(node.outTensors.size());
        node.torchTensors.resize(node.outTensors.size());
    }
    InitTensorType();
}

void Model::Graph::InitTensorType()
{
    for (auto &node : nodes) {
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            node.inTensorTypes.at(i) =
                IsInternalTensor(node.inTensors.at(i)) ? Model::INTERMEDIATE_TENSOR : Model::NOT_INTERMEDIATE_TENSOR;
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            node.outTensorTypes.at(i) =
                IsInternalTensor(node.outTensors.at(i)) ? Model::INTERMEDIATE_TENSOR : Model::NOT_INTERMEDIATE_TENSOR;
        }
    }
}

bool Model::Graph::IsInternalTensor(const AsdOps::Tensor *tensor)
{
    for (auto &internalTensor : internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }

    return false;
}

Model::Model(const std::string &modelName, const std::string &param) : modelName_(modelName), param_(param)
{
    AsdRtDeviceGetCurrent(&currentDevId_);

    const char *envStr = std::getenv("TASK_QUEUE_ENABLE");
    isTaskQueueEnable_ = (envStr != nullptr && std::string(envStr) == "1") ? true : false;

    envStr = std::getenv("ACLTRANSFORMER_PLAN_EXECUTE_ASYNC");
    isUsePlanExecuteAsync_ = (envStr != nullptr && std::string(envStr) == "1") ? true : false;
    if (isUsePlanExecuteAsync_ && !isTaskQueueEnable_) {
        std::thread thread = std::thread(std::bind(&Model::ThreadProcessTask, this));
        taskProcessThread_ = std::move(thread);
    }

    ASD_LOG(FATAL) << modelName_ << " new, isTaskQueueEnable:" << isTaskQueueEnable_
                   << ", isUsePlanExecuteAsync:" << isUsePlanExecuteAsync_ << ", currentDevId:" << currentDevId_;
}

Model::~Model() {}

void Model::Init()
{
    BuildGraph();
    graph_.Init();
    ASD_LOG(INFO) << modelName_ << " init graph:\n" << graph_.ToString();
}

void Model::SetWeight(const std::vector<AsdOps::Tensor> &weightTensors)
{
    if (graph_.weightTensors.size() != weightTensors.size()) {
        ASD_LOG(ERROR) << modelName_ << " weightTensors.size:" << weightTensors.size() << " != "
                       << " graph.weightTensors.size:" << graph_.weightTensors.size();
        return;
    }

    graph_.weightTensors = weightTensors;
}

AsdOps::Status Model::Execute(Handle handle, std::vector<AsdOps::Tensor> &inTensors,
                              std::vector<AsdOps::Tensor> &outTensors, const std::string &param)
{
    if (graph_.inTensors.size() != inTensors.size() || graph_.outTensors.size() != outTensors.size()) {
        ASD_LOG(ERROR) << modelName_ << " graph.inTensors.size:" << graph_.inTensors.size()
                       << ", inTensors.size:" << inTensors.size()
                       << ", graph.outTensors.size:" << graph_.outTensors.size()
                       << ", outTensors.size:" << outTensors.size();
        return AsdOps::Status::FailStatus(1, "invalid inTensors or outtensors size");
    }

    timer_.Reset();
    allTaskFinish_ = false;
    handle_ = handle;
    graph_.inTensors = inTensors;
    graph_.outTensors = outTensors;
    ASD_LOG(INFO) << modelName_ << " execute start, executeCount:" << executeCount_ << ", graph:\n"
                  << graph_.ToString();

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); ++nodeId) {
        auto &node = graph_.nodes.at(nodeId);
        ParseVarintPackParam(param, nodeId, node.variantPack.param);
        BuildNodeVariantPack(nodeId);
        ExecuteNode(nodeId);
    }

    WaitAsyncPlanExecuteFinish();

    AsdOps::GetSingleton<Statistic>().totalTime += timer_.ElapsedMicroSecond();
    ASD_LOG(FATAL) << modelName_ << " executeCount:" << executeCount_ << ", statistic:["
                   << AsdOps::GetSingleton<Statistic>().ToString() << "]";
    AsdOps::GetSingleton<Statistic>().Reset();

    executeCount_++;

    return AsdOps::Status::OkStatus();
}

AsdOps::Status Model::ParseVarintPackParam(const std::string &param, int nodeId, AsdOps::Any &variantPackParam)
{
    return AsdOps::Status::OkStatus();
}

void Model::BuildNodeVariantPack(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);

    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
        ASD_LOG(INFO) << modelName_ << " nodes[" << nodeId << "] inTensors[" << i
                      << "]:" << TensorUtil::AsdOpsTensorToString(node.variantPack.inTensors.at(i));
    }

    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    AsdOps::Status st = node.operation->InferShape(node.variantPack.inTensors, outTensorDescs);
    ASD_LOG_IF(!st.Ok(), FATAL) << modelName_ << " nodes[" << nodeId << "] " << node.operation->GetName()
                                << " infer shape fail, error:" << st.Message();
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ASD_LOG(INFO) << modelName_ << " nodes[" << nodeId << "] outTensorDescs[" << i
                      << "]:" << TensorUtil::AsdOpsTensorDescToString(outTensorDescs.at(i));
    }

    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
        if (node.outTensorTypes.at(i) == Model::INTERMEDIATE_TENSOR) {
            if (node.torchTensors.at(i).numel() != outTensorDescs.at(i).Numel()) {
                ASD_LOG(INFO) << modelName_ << "  nodes[" << nodeId << "] new outtensors[" << i << "]";
                AsdOps::Timer timer;
                node.torchTensors.at(i) = Utils::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(i));
                AsdOps::GetSingleton<Statistic>().createTensorTime += timer.ElapsedMicroSecond();
            }
            node.variantPack.outTensors.at(i) = Utils::AtTensor2AsdTensor(node.torchTensors.at(i));
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
        }
        if (node.variantPack.outTensors.at(i).desc.dims != outTensorDescs.at(i).dims) {
            ASD_LOG(FATAL) << modelName_ << "  nodes[" << nodeId << "] new outTensorDescs[" << i
                           << "]:" << TensorUtil::AsdOpsTensorDescToString(outTensorDescs.at(i))
                           << ", node.variantPack.outTensors.at[" << i
                           << "].desc:" << TensorUtil::AsdOpsTensorDescToString(node.variantPack.outTensors.at(i).desc);
        }
    }
}

void Model::ExecuteNode(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);

    Plan &plan = *node.plan;

    AsdOps::Timer timer;
    AsdOps::Status st = plan.Setup(handle_, node.variantPack);
    AsdOps::GetSingleton<Statistic>().planSetupTime += timer.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << modelName_ << " setup plan[" << nodeId << "] fail, not call execute";
        return;
    }

    node.variantPack.workspaceSize = plan.GetWorkspaceSize();
    ASD_LOG(INFO) << modelName_ << " get plan[" << nodeId << "] workspace size:" << node.variantPack.workspaceSize;

    if (node.variantPack.workspaceSize > 0) {
        node.variantPack.workspace = AsdOps::GetSingleton<Context>().GetWorkspaceBuffer(node.variantPack.workspaceSize);
    }

    if (isUsePlanExecuteAsync_) {
        AsdOps::Timer timer;
        ExecutePlanAsync(nodeId);
        AsdOps::GetSingleton<Statistic>().planAsyncTime += timer.ElapsedMicroSecond();
    } else {
        ExecutePlanSync(nodeId);
    }
}

void Model::ThreadProcessTask()
{
    ASD_LOG(FATAL) << modelName_ << " thread process operations start";
    int ret = AsdRtDeviceSetCurrent(currentDevId_);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;

    size_t processTaskCount = 0;
    while (true) {
        int nodeId = PopTask();
        ExecutePlanSync(nodeId);
        processTaskCount++;
        if (processTaskCount == graph_.nodes.size()) {
            ASD_LOG(INFO) << modelName_ << " thread process all operations";
            processTaskCount = 0;
            allTaskFinish_ = true;
        }
    }
}

void Model::ExecutePlanSync(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);
    Plan &plan = *node.plan;
    VariantPack &variantPack = node.variantPack;

    if (AsdOps::GetSingleton<Config>().IsSaveTensor()) {
        std::string dir = GetSaveTensorDir() + "/" + std::to_string(nodeId) + "_";
        plan.SetRunnerSaveTensorDir(dir);
    }

    ASD_LOG(INFO) << modelName_ << " execute plan[" << nodeId << "] start";
    AsdOps::Timer timer;
    AsdOps::Status st = plan.Execute(handle_, variantPack);
    AsdOps::GetSingleton<Statistic>().planExecuteTime += timer.ElapsedMicroSecond();
    ASD_LOG_IF(!st.Ok(), ERROR) << "  execute plan[" << nodeId << "] fail, error:" << st.Message();
}

void Model::ExecutePlanAsync(int nodeId)
{
    if (isTaskQueueEnable_) {
#ifdef TORCH_SETCUSTOMHANDLER
        at_npu::native::OpCommand cmd;
        cmd.Name(modelName_ + std::to_string(nodeId));
        cmd.SetCustomHandler([=]() {
            ExecutePlanSync(nodeId);
            return 0;
        });
        cmd.Run();
#else
        ASD_LOG(FATAL) << modelName_ << " torch_npu is low, can't support SetCustomHandler";
#endif
    } else {
        PushTask(nodeId);
    }
}

void Model::PushTask(int nodeId)
{
    std::unique_lock<std::mutex> lock(mutex_);
    taskQueue_.push(nodeId);
    lock.unlock();
    cond_.notify_one();
}

int Model::PopTask()
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (taskQueue_.empty()) {
        cond_.wait(lock);
    }
    int nodeId = taskQueue_.front();
    taskQueue_.pop();
    return nodeId;
}

void Model::WaitAsyncPlanExecuteFinish()
{
    if (isUsePlanExecuteAsync_ && !isTaskQueueEnable_) {
        while (true) {
            if (allTaskFinish_) {
                ASD_LOG(INFO) << modelName_ << " allTaskFinish is true, break";
                break;
            }
        }
    }
}

std::string Model::GetSaveTensorDir()
{
    std::string dir = std::to_string(executeCount_) + "/0_Model";
    return Config::GetSaveTensorDir() + "/" + dir;
}
} // namespace AclTransformer