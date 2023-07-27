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
#include "chatglm6bmodel_decoder_quant_flash_torch.h"
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
#include "torch/context/context.h"
#include "models/chatglm6b/chatglm6blayer_decoder_quant_flash_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_first_quant_flash_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_last_quant_flash_operation.h"
const size_t WEIGHT_COUNT_PER_LAYER = 16;

ChatGlm6BModelDecoderQuantFlashTorch::ChatGlm6BModelDecoderQuantFlashTorch()
{
    AsdRtDeviceGetCurrent(&currentDevId_);

    const char *envStr = std::getenv("TASK_QUEUE_ENABLE");
    isTaskQueueEnable_ = (envStr != nullptr && std::string(envStr) == "1") ? true : false;

    envStr = std::getenv("ACLTRANSFORMER_PLAN_EXECUTE_ASYNC");
    isUsePlanExecuteAsync_ = (envStr != nullptr && std::string(envStr) == "1") ? true : false;
    if (isUsePlanExecuteAsync_ && !isTaskQueueEnable_) {
        std::thread thread = std::thread(std::bind(&ChatGlm6BModelDecoderQuantFlashTorch::ThreadProcessTask, this));
        taskProcessThread_ = std::move(thread);
    }

    ASD_LOG(FATAL) << "ChatGlm6BModelDecoderQuantFlashTorch new, TASK_QUEUE_ENABLE:" << isTaskQueueEnable_
                   << ", ACLTRANSFORMER_PLAN_EXECUTE_ASYNC:" << isUsePlanExecuteAsync_
                   << ", currentDevId:" << currentDevId_;
}

ChatGlm6BModelDecoderQuantFlashTorch::~ChatGlm6BModelDecoderQuantFlashTorch() {}

void ChatGlm6BModelDecoderQuantFlashTorch::SetParam(std::string param)
{
    ASD_LOG(INFO) << "ChatGlm6BModel set param start, param:" << param;
    modelParam_.FromString(param);
    operations_.resize(modelParam_.layerNum);
    plans_.resize(modelParam_.layerNum);
    outTensors_.resize(modelParam_.layerNum);
    variantPacks_.resize(modelParam_.layerNum);

    for (int i = 0; i < modelParam_.layerNum; ++i) {
        AclTransformer::ChatGlm6BLayerQuantFlashParam opParam;
        opParam.layerNormEps = modelParam_.layerNormEps;
        opParam.headNum = modelParam_.headNum;
        opParam.transKey = modelParam_.transKey;
        opParam.dk = modelParam_.dk;
        opParam.layerId = i;
        opParam.residualAddScale = modelParam_.residualAddScale;

        opParam.tokenOffset = modelParam_.tokenOffset;
        opParam.seqLen = modelParam_.seqLen;

        opParam.residualAddScale = modelParam_.residualAddScale;
        opParam.qkvInputScale = modelParam_.qkvInputScale[i];
        opParam.qkvInputOffset = modelParam_.qkvInputOffset[i];
        opParam.denseInputScale = modelParam_.denseInputScale[i];
        opParam.denseInputOffset = modelParam_.denseInputOffset[i];
        opParam.selfLnInputScale = modelParam_.selfLnInputScale[i];
        opParam.selfLnInputOffset = modelParam_.selfLnInputOffset[i];
        opParam.ffnOutInputScale = modelParam_.ffnOutInputScale[i];
        opParam.ffnOutInputOffset = modelParam_.ffnOutInputOffset[i];
        AclTransformer::Operation *op;
        if (i == 0) {
            op = new AclTransformer::ChatGlm6BLayerDecoderFirstQuantFlashOperation(opParam);
        } else if (i == modelParam_.layerNum - 1) {
            op = new AclTransformer::ChatGlm6BLayerDecoderLastQuantFlashOperation(opParam);
        } else {
            op = new AclTransformer::ChatGlm6BLayerDecoderQuantFlashOperation(opParam);
        }

        operations_.at(i).reset(op);
        AclTransformer::Plan *plan = new AclTransformer::Plan();
        op->BuildPlan(plan);
        plans_.at(i).reset(plan);
    }

    ASD_LOG(INFO) << "ChatGlm6BModel set param end";
}

void ChatGlm6BModelDecoderQuantFlashTorch::SetWeight(std::vector<torch::Tensor> weightTensors)
{
    if (weightTensors.size() != modelParam_.layerNum * WEIGHT_COUNT_PER_LAYER) {
        ASD_LOG(ERROR) << "ChatGlm6BModel set weight fail, weightTensors.size:" << weightTensors.size()
                       << " != " << modelParam_.layerNum * WEIGHT_COUNT_PER_LAYER;
        return;
    }

    ASD_LOG(INFO) << "ChatGlm6BModel set weight success, size:" << weightTensors.size();

    weightTensors_ = weightTensors;
    Utils::ContiguousAtTensor(weightTensors_);
}

std::vector<torch::Tensor> ChatGlm6BModelDecoderQuantFlashTorch::Execute(
    torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor, torch::Tensor cosTableTensor,
    torch::Tensor sinTableTensor, torch::Tensor attentionMaskTensor, torch::Tensor pastKeyTensors,
    torch::Tensor pastValueTensors, torch::Tensor tokenOffset, torch::Tensor seqLen,
    std::vector<torch::Tensor> layerIdInput, std::string param)
{
    timer_.Reset();
    if (plans_.empty()) {
        SetParam(param);
    }
    torch::Tensor outTensor;
    torch::Tensor presendKeyTensors;
    torch::Tensor presentValueTensors;

    ExecuteOutImpl(hiddenStateTensor, positionIdTensor, cosTableTensor, sinTableTensor, attentionMaskTensor,
                   pastKeyTensors, pastValueTensors, tokenOffset, seqLen, layerIdInput, outTensor, true, param);

    std::vector<torch::Tensor> outTensors(1);
    size_t tensorId = 0;
    outTensors.at(tensorId) = outTensor;
    return outTensors;
}

void ChatGlm6BModelDecoderQuantFlashTorch::ExecuteOut(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor,
                                                      torch::Tensor cosTableTensor, torch::Tensor sinTableTensor,
                                                      torch::Tensor attentionMaskTensor, torch::Tensor pastKeyTensors,
                                                      torch::Tensor pastValueTensors, torch::Tensor tokenOffset,
                                                      torch::Tensor seqLen, std::vector<torch::Tensor> layerIdInput,
                                                      torch::Tensor outTensor, std::string param)
{
    timer_.Reset();
    if (plans_.empty()) {
        SetParam(param);
    }
    ExecuteOutImpl(hiddenStateTensor, positionIdTensor, cosTableTensor, sinTableTensor, attentionMaskTensor,
                   pastKeyTensors, pastValueTensors, tokenOffset, seqLen, layerIdInput, outTensor, false, param);
}

void ChatGlm6BModelDecoderQuantFlashTorch::ExecuteOutImpl(
    torch::Tensor &hiddenStateTensor, torch::Tensor &positionIdTensor, torch::Tensor &cosTableTensor,
    torch::Tensor &sinTableTensor, torch::Tensor &attentionMaskTensor, torch::Tensor &pastKeyTensors,
    torch::Tensor &pastValueTensors, torch::Tensor &tokenOffset, torch::Tensor &seqLen,
    std::vector<torch::Tensor> &layerIdInput, torch::Tensor &outTensor, bool newOut, std::string param)
{
    handle_ = {Utils::GetCurrentStream()};
    ParseParam(param);
    allTaskFinish_ = false;

    Utils::ContiguousAtTensor(hiddenStateTensor);
    Utils::ContiguousAtTensor(positionIdTensor);
    Utils::ContiguousAtTensor(cosTableTensor);
    Utils::ContiguousAtTensor(sinTableTensor);
    Utils::ContiguousAtTensor(attentionMaskTensor);
    Utils::ContiguousAtTensor(pastKeyTensors);
    Utils::ContiguousAtTensor(pastValueTensors);
    Utils::ContiguousAtTensor(tokenOffset);
    Utils::ContiguousAtTensor(seqLen);
    Utils::ContiguousAtTensor(layerIdInput);

    if (!newOut) {
        Utils::ContiguousAtTensor(outTensor);
    }

    AclTransformer::Operation *operation = operations_.at(0).get();
    std::vector<torch::Tensor> opAtInTensors(operation->GetInTensorCount());
    std::vector<torch::Tensor> opAtOutTensors(operation->GetOutTensorCount());

    torch::Tensor firstInTensor = hiddenStateTensor;
    torch::Tensor firstResInTensor = hiddenStateTensor;

    for (int layerId = 0; layerId < modelParam_.layerNum - 1; ++layerId) {
        size_t inTensorId = 0;
        opAtInTensors.at(inTensorId++) = firstInTensor;

        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            opAtInTensors.at(inTensorId++) = weightTensors_.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        opAtInTensors.at(inTensorId++) = positionIdTensor;    // positionIdTensor
        opAtInTensors.at(inTensorId++) = cosTableTensor;      // cosTable
        opAtInTensors.at(inTensorId++) = sinTableTensor;      // sinTable
        opAtInTensors.at(inTensorId++) = attentionMaskTensor; // attentionMaskTensor
        opAtInTensors.at(inTensorId++) = pastKeyTensors;
        opAtInTensors.at(inTensorId++) = pastValueTensors;
        opAtInTensors.at(inTensorId++) = tokenOffset;
        opAtInTensors.at(inTensorId++) = seqLen; // seqLen
        opAtInTensors.at(inTensorId++) = layerIdInput.at(layerId);
        opAtInTensors.at(inTensorId++) = firstResInTensor;
        ExecuteLayerOperation(layerId, opAtInTensors, outTensor, firstResInTensor, newOut);
        firstInTensor = outTensor;
    }

    size_t lastInTensorId = 0;
    size_t lastLayerId = modelParam_.layerNum - 1;
    opAtInTensors.at(lastInTensorId++) = firstInTensor;
    for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
        opAtInTensors.at(lastInTensorId++) = weightTensors_.at(lastLayerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
    }
    opAtInTensors.at(lastInTensorId++) = positionIdTensor;    // positionIdTensor
    opAtInTensors.at(lastInTensorId++) = cosTableTensor;      // cosTable
    opAtInTensors.at(lastInTensorId++) = sinTableTensor;      // sinTable
    opAtInTensors.at(lastInTensorId++) = attentionMaskTensor; // attentionMaskTensor
    opAtInTensors.at(lastInTensorId++) = pastKeyTensors;
    opAtInTensors.at(lastInTensorId++) = pastValueTensors;
    opAtInTensors.at(lastInTensorId++) = tokenOffset;
    opAtInTensors.at(lastInTensorId++) = seqLen; // seqLen
    opAtInTensors.at(lastInTensorId++) = layerIdInput.at(lastLayerId);
    opAtInTensors.at(lastInTensorId++) = firstResInTensor;
    ExecuteLastLayerOperation(lastLayerId, opAtInTensors, outTensor, newOut);

    WaitAsyncPlanExecuteFinish();
    AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime += timer_.ElapsedMicroSecond();
    ASD_LOG(FATAL) << "ChatGlm6BModel executeCount:" << executeCount_ << ", statistic:["
                   << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString() << "]";
    AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();

    executeCount_++;
}

void ChatGlm6BModelDecoderQuantFlashTorch::BuildVariantPack(int layerId, std::vector<torch::Tensor> &atInTensors,
                                                            torch::Tensor &outTensor, torch::Tensor &resIn, bool newOut,
                                                            AclTransformer::VariantPack &variantPack)
{
    variantPack.inTensors.clear();
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << layerId << " atInTensors[" << i
                      << "].options:" << atInTensors.at(i).options() << ", data:" << atInTensors.at(i).data_ptr()
                      << ", storage_offset:" << atInTensors.at(i).storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atInTensors.at(i));
        variantPack.inTensors.push_back(Utils::AtTensor2AsdTensor(atInTensors.at(i)));
    }

    if (newOut) {
        AclTransformer::Operation *operation = operations_.at(layerId).get();
        AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
        operation->InferShape(variantPack.inTensors, outTensorDescs);
        outTensor = Utils::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(0));
        resIn = Utils::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(1));
    }
    variantPack.outTensors.clear();
    variantPack.outTensors.push_back(Utils::AtTensor2AsdTensor(outTensor));
    variantPack.outTensors.push_back(Utils::AtTensor2AsdTensor(resIn));
}

void ChatGlm6BModelDecoderQuantFlashTorch::ExecuteLayerOperation(int layerId, std::vector<torch::Tensor> &opAtInTensors,
                                                                 torch::Tensor &outTensor, torch::Tensor &resIn,
                                                                 bool newOut)
{
    AclTransformer::Plan &plan = *plans_.at(layerId);

    AclTransformer::VariantPack &variantPack = variantPacks_.at(layerId);
    variantPack.param = variantPackParam_;
    BuildVariantPack(layerId, opAtInTensors, outTensor, resIn, newOut, variantPack);

    AsdOps::Timer timer1;
    AsdOps::Status st = plan.Setup(handle_, variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime += timer1.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "ChatGlm6BModelLayer_" << layerId << " setup plan fail, not call execute";
        return;
    }

    variantPack.workspaceSize = plan.GetWorkspaceSize();
    ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << layerId << " get plan workspace size:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        variantPack.workspace =
            AsdOps::GetSingleton<AclTransformer::Context>().GetWorkspaceBuffer(variantPack.workspaceSize);
    }

    if (isUsePlanExecuteAsync_) {
        if (isTaskQueueEnable_) {
#ifdef TORCH_SETCUSTOMHANDLER
            at_npu::native::OpCommand cmd;
            cmd.Name("ChatGlm6BModelLayer_" + std::to_string(layerId));
            cmd.SetCustomHandler([=]() {
                ExecutePlan(layerId);
                return 0;
            });
            cmd.Run();
#else
            ASD_LOG(FATAL) << "ChatGlm6BModelLayer_" << layerId << " torch_npu is low, can't support SetCustomHandler";
#endif
        } else {
            PushTask(layerId);
        }
    } else {
        ExecutePlan(layerId);
    }
}

void ChatGlm6BModelDecoderQuantFlashTorch::BuildVariantPackLast(int layerId, std::vector<torch::Tensor> &atInTensors,
                                                                torch::Tensor &outTensor, bool newOut,
                                                                AclTransformer::VariantPack &variantPack)
{
    variantPack.inTensors.clear();
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << layerId << " atInTensors[" << i
                      << "].options:" << atInTensors.at(i).options() << ", data:" << atInTensors.at(i).data_ptr()
                      << ", storage_offset:" << atInTensors.at(i).storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atInTensors.at(i));
        variantPack.inTensors.push_back(Utils::AtTensor2AsdTensor(atInTensors.at(i)));
    }

    if (newOut) {
        AclTransformer::Operation *operation = operations_.at(layerId).get();
        AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
        operation->InferShape(variantPack.inTensors, outTensorDescs);
        outTensor = Utils::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(0));
    }

    variantPack.outTensors.clear();
    variantPack.outTensors.push_back(Utils::AtTensor2AsdTensor(outTensor));
}

void ChatGlm6BModelDecoderQuantFlashTorch::ExecuteLastLayerOperation(int layerId,
                                                                     std::vector<torch::Tensor> &opAtInTensors,
                                                                     torch::Tensor &outTensor, bool newOut)
{
    AclTransformer::Plan &plan = *plans_.at(layerId);

    AclTransformer::VariantPack &variantPack = variantPacks_.at(layerId);
    variantPack.param = variantPackParam_;
    BuildVariantPackLast(layerId, opAtInTensors, outTensor, newOut, variantPack);

    AsdOps::Timer timer1;
    AsdOps::Status st = plan.Setup(handle_, variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime += timer1.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "ChatGlm6BModelLayer_" << layerId << " setup plan fail, not call execute";
        return;
    }

    variantPack.workspaceSize = plan.GetWorkspaceSize();
    ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << layerId << " get plan workspace size:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        variantPack.workspace =
            AsdOps::GetSingleton<AclTransformer::Context>().GetWorkspaceBuffer(variantPack.workspaceSize);
    }

    if (isUsePlanExecuteAsync_) {
        if (isTaskQueueEnable_) {
#ifdef TORCH_SETCUSTOMHANDLER
            at_npu::native::OpCommand cmd;
            cmd.Name("ChatGlm6BModelLayer_" + std::to_string(layerId));
            cmd.SetCustomHandler([=]() {
                ExecutePlan(layerId);
                return 0;
            });
            cmd.Run();
#else
            ASD_LOG(FATAL) << "ChatGlm6BModelLayer_" << layerId << " torch_npu is low, can't support SetCustomHandler";
#endif
        } else {
            PushTask(layerId);
        }
    } else {
        ExecutePlan(layerId);
    }
}

void ChatGlm6BModelDecoderQuantFlashTorch::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    variantPackParam_.tokenOffset.clear();
    for (auto item : paramJson["tokenOffset"]) {
        variantPackParam_.tokenOffset.push_back(item.get<int>());
    }
    variantPackParam_.seqLen.clear();
    for (auto item : paramJson["seqLen"]) {
        variantPackParam_.seqLen.push_back(item.get<int>());
    }
}

void ChatGlm6BModelDecoderQuantFlashTorch::ThreadProcessTask()
{
    ASD_LOG(FATAL) << "ChatGlm6BModelLayer ThreadProcessTask start";
    int ret = AsdRtDeviceSetCurrent(currentDevId_);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;

    int processTaskCount = 0;
    while (true) {
        int layerId = PopTask();
        ExecutePlan(layerId);
        processTaskCount++;
        if (processTaskCount == modelParam_.layerNum) {
            ASD_LOG(INFO) << "ChatGlm6BModelLayer thread process all layers";
            processTaskCount = 0;
            allTaskFinish_ = true;
        }
    }
}

void ChatGlm6BModelDecoderQuantFlashTorch::ExecutePlan(int layerId)
{
    AsdOps::Timer timer2;
    AclTransformer::Plan &plan = *plans_.at(layerId);
    AclTransformer::VariantPack &variantPack = variantPacks_.at(layerId);
    ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << layerId << " execute plan start";
    AsdOps::Status st = plan.Execute(handle_, variantPack);
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
        AsdRtStreamSynchronize(handle_.stream);
        std::string dirPath =
            AclTransformer::Config::GetSaveTensorDir() + "/ChatGlm6BModelLayer_" + std::to_string(layerId);
        AclTransformer::TensorUtil::SaveVariantPack(handle_, variantPack, dirPath);
        ASD_LOG(FATAL) << "ChatGlm6BModelLayer_" << layerId << " save variant pack, dir:" << dirPath;
    }

    AsdOps::GetSingleton<AclTransformer::Statistic>().planExecuteTime += timer2.ElapsedMicroSecond();
    ASD_LOG_IF(!st.Ok(), ERROR) << "ChatGlm6BModelLayer_" << layerId << " execute plan fail, error:" << st.Message();
}

void ChatGlm6BModelDecoderQuantFlashTorch::PushTask(int layerId)
{
    std::unique_lock<std::mutex> lock(mutex_);
    taskQueue_.push(layerId);
    lock.unlock();
    cond_.notify_one();
}

int ChatGlm6BModelDecoderQuantFlashTorch::PopTask()
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (taskQueue_.empty()) {
        cond_.wait(lock);
    }
    int layerId = taskQueue_.front();
    taskQueue_.pop();
    return layerId;
}

void ChatGlm6BModelDecoderQuantFlashTorch::WaitAsyncPlanExecuteFinish()
{
    if (isUsePlanExecuteAsync_ && !isTaskQueueEnable_) {
        while (true) {
            if (allTaskFinish_) {
                ASD_LOG(INFO) << "ChatGlm6BModel allTaskFinish is true, break";
                break;
            }
        }
    }
}

TORCH_LIBRARY(ChatGlm6BModelDecoderQuantFlashTorch, m)
{
    m.class_<ChatGlm6BModelDecoderQuantFlashTorch>("ChatGlm6BModelDecoderQuantFlashTorch")
        .def(torch::init<>())
        .def("set_param", &ChatGlm6BModelDecoderQuantFlashTorch::SetParam)
        .def("set_weight", &ChatGlm6BModelDecoderQuantFlashTorch::SetWeight)
        .def("execute", &ChatGlm6BModelDecoderQuantFlashTorch::Execute)
        .def("execute_out", &ChatGlm6BModelDecoderQuantFlashTorch::ExecuteOut);
}