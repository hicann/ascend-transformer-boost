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
#include "chatglm6bmodel_decoder_all_torch.h"
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
#include "acltransformer/ops/embedding_operation.h"
#include "acltransformer/ops/transpose_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "models/chatglm6b/chatglm6blayer_decoder_flashattention_operation.h"

const size_t WEIGHT_COUNT_PER_LAYER = 12;
const int WEIGHT_OFFSET = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 2;
const int OPERATION_COUNT_AFTER_LAYER = 1;

ChatGlm6BModelDecoderAllTorch::ChatGlm6BModelDecoderAllTorch()
{
    std::vector<AsdOps::Operation *> ops;
    AsdOps::Ops::Instance().GetAllOperations(ops);

    AsdRtDeviceGetCurrent(&currentDevId_);

    const char *envStr = std::getenv("TASK_QUEUE_ENABLE");
    isTaskQueueEnable_ = (envStr != nullptr && std::string(envStr) == "1") ? true : false;

    envStr = std::getenv("ACLTRANSFORMER_PLAN_EXECUTE_ASYNC");
    isUsePlanExecuteAsync_ = (envStr != nullptr && std::string(envStr) == "1") ? true : false;
    if (isUsePlanExecuteAsync_ && !isTaskQueueEnable_) {
        std::thread thread = std::thread(std::bind(&ChatGlm6BModelDecoderAllTorch::ThreadProcessTask, this));
        taskProcessThread_ = std::move(thread);
    }

    ASD_LOG(FATAL) << "ChatGlm6BModelDecoderAllTorch new, TASK_QUEUE_ENABLE:" << isTaskQueueEnable_
                   << ", ACLTRANSFORMER_PLAN_EXECUTE_ASYNC:" << isUsePlanExecuteAsync_
                   << ", currentDevId:" << currentDevId_;
}

ChatGlm6BModelDecoderAllTorch::~ChatGlm6BModelDecoderAllTorch() {}

void ChatGlm6BModelDecoderAllTorch::SetParam(std::string param)
{
    ASD_LOG(INFO) << "ChatGlm6BModel set param start, param:" << param;
    modelParam_.FromString(param);
    int operationNum = modelParam_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    operations_.resize(operationNum);
    plans_.resize(operationNum);
    outTensors_.resize(operationNum);
    variantPacks_.resize(operationNum);

    AclTransformer::EmbeddingParam wordEmbeddingParam;
    AclTransformer::TransposeParam TransposeParam = {{1, 0, 2}};

    AclTransformer::Operation *wordEmbedding = new AclTransformer::EmbeddingOperation(wordEmbeddingParam);
    AclTransformer::Plan *wordEmbeddingPlan = new AclTransformer::Plan();
    wordEmbedding->BuildPlan(wordEmbeddingPlan);
    operations_.at(0).reset(wordEmbedding);
    plans_.at(0).reset(wordEmbeddingPlan);

    AclTransformer::Operation *transpose = new AclTransformer::TransposeOperation(TransposeParam);
    AclTransformer::Plan *transposePlan = new AclTransformer::Plan();
    transpose->BuildPlan(transposePlan);
    operations_.at(1).reset(transpose);
    plans_.at(1).reset(transposePlan);

    for (int i = 0; i < modelParam_.layerNum; ++i) {
        AclTransformer::ChatGlm6BLayerDecoderFlashAttentionParam opParam;
        opParam.layerNormEps = modelParam_.layerNormEps;
        opParam.headNum = modelParam_.headNum;
        opParam.transKey = modelParam_.transKey;
        opParam.dk = modelParam_.dk;
        opParam.layerId = i;
        opParam.residualAddScale = modelParam_.residualAddScale;

        opParam.tokenOffset = modelParam_.tokenOffset;
        opParam.seqLen = modelParam_.seqLen;

        AclTransformer::Operation *op = new AclTransformer::ChatGlm6BLayerDecoderFlashAttentionOperation(opParam);

        operations_.at(i + OPERATION_COUNT_BEFORE_LAYER).reset(op);
        AclTransformer::Plan *plan = new AclTransformer::Plan();
        op->BuildPlan(plan);
        plans_.at(i + OPERATION_COUNT_BEFORE_LAYER).reset(plan);
    }

    AclTransformer::NormParam finalNormParam = {modelParam_.layerNormEps};
    AclTransformer::Operation *finalLayernorm = new AclTransformer::NormOperation(finalNormParam);
    AclTransformer::Plan *finalLayernormPlan = new AclTransformer::Plan();
    finalLayernorm->BuildPlan(finalLayernormPlan);
    operations_.at(modelParam_.layerNum + OPERATION_COUNT_BEFORE_LAYER).reset(finalLayernorm);
    plans_.at(modelParam_.layerNum + OPERATION_COUNT_BEFORE_LAYER).reset(finalLayernormPlan);

    ASD_LOG(INFO) << "ChatGlm6BModel set param end";
}

void ChatGlm6BModelDecoderAllTorch::SetWeight(std::vector<torch::Tensor> weightTensors)
{
    if (weightTensors.size() != modelParam_.layerNum * WEIGHT_COUNT_PER_LAYER + 3) {
        ASD_LOG(ERROR) << "ChatGlm6BModel set weight fail, weightTensors.size:" << weightTensors.size()
                       << " != " << modelParam_.layerNum * WEIGHT_COUNT_PER_LAYER;
        return;
    }

    ASD_LOG(INFO) << "ChatGlm6BModel set weight success, size:" << weightTensors.size();

    weightTensors_ = weightTensors;
    Utils::ContiguousAtTensor(weightTensors_);
}

std::vector<torch::Tensor> ChatGlm6BModelDecoderAllTorch::Execute(
    torch::Tensor inputIds, torch::Tensor positionIdTensor, torch::Tensor cosTableTensor, torch::Tensor sinTableTensor,
    torch::Tensor attentionMaskTensor, torch::Tensor pastKeyTensors, torch::Tensor pastValueTensors,
    torch::Tensor tokenOffset, torch::Tensor seqLen, std::vector<torch::Tensor> layerIdInput, std::string param)
{
    timer_.Reset();
    if (plans_.empty()) {
        SetParam(param);
    }
    torch::Tensor outTensor;
    torch::Tensor presendKeyTensors;
    torch::Tensor presentValueTensors;

    ExecuteOutImpl(inputIds, positionIdTensor, cosTableTensor, sinTableTensor, attentionMaskTensor, pastKeyTensors,
                   pastValueTensors, tokenOffset, seqLen, layerIdInput, outTensor, true, param);

    std::vector<torch::Tensor> outTensors(1);
    size_t tensorId = 0;
    outTensors.at(tensorId) = outTensor;
    return outTensors;
}

void ChatGlm6BModelDecoderAllTorch::ExecuteOut(torch::Tensor inputIds, torch::Tensor positionIdTensor,
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
    ExecuteOutImpl(inputIds, positionIdTensor, cosTableTensor, sinTableTensor, attentionMaskTensor, pastKeyTensors,
                   pastValueTensors, tokenOffset, seqLen, layerIdInput, outTensor, false, param);
}

void ChatGlm6BModelDecoderAllTorch::ExecuteOutImpl(torch::Tensor &inputIds, torch::Tensor &positionIdTensor,
                                                   torch::Tensor &cosTableTensor, torch::Tensor &sinTableTensor,
                                                   torch::Tensor &attentionMaskTensor, torch::Tensor &pastKeyTensors,
                                                   torch::Tensor &pastValueTensors, torch::Tensor &tokenOffset,
                                                   torch::Tensor &seqLen, std::vector<torch::Tensor> &layerIdInput,
                                                   torch::Tensor &outTensor, bool newOut, std::string param)
{
    handle_ = {Utils::GetCurrentStream()};
    ParseParam(param);
    allTaskFinish_ = false;

    if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
        if (executeCount_ >= AsdOps::GetSingleton<AclTransformer::Config>().GetSaveTensorMaxNum()) {
            AsdOps::GetSingleton<AclTransformer::Config>().DisableSaveTensor();
        }
    }

    Utils::ContiguousAtTensor(inputIds);
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

    AclTransformer::Operation *wordEmbedding = operations_.at(0).get();
    std::vector<torch::Tensor> wordEmbeddingAtInTensors(wordEmbedding->GetInTensorCount());
    torch::Tensor hiddenStateTensorPre;

    wordEmbeddingAtInTensors.at(0) = weightTensors_.at(0);
    wordEmbeddingAtInTensors.at(1) = inputIds;

    ExecuteOperation(0, wordEmbeddingAtInTensors, hiddenStateTensorPre, newOut);

    AclTransformer::Operation *transpose = operations_.at(1).get();
    std::vector<torch::Tensor> transposeAtInTensors(transpose->GetInTensorCount());
    torch::Tensor hiddenStateTensor;

    transposeAtInTensors.at(0) = hiddenStateTensorPre;

    ExecuteOperation(1, transposeAtInTensors, hiddenStateTensor, newOut);

    AclTransformer::Operation *layerOperation = operations_.at(2).get();
    std::vector<torch::Tensor> opAtInTensors(layerOperation->GetInTensorCount());

    torch::Tensor firstInTensor = hiddenStateTensor;
    for (int layerId = 0; layerId < modelParam_.layerNum; ++layerId) {
        size_t inTensorId = 0;
        opAtInTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            opAtInTensors.at(inTensorId++) =
                weightTensors_.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_OFFSET);
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

        ExecuteOperation(layerId + OPERATION_COUNT_BEFORE_LAYER, opAtInTensors, outTensor, newOut);

        firstInTensor = outTensor;
    }

    const int finalLayerNormId = modelParam_.layerNum + OPERATION_COUNT_BEFORE_LAYER;
    AclTransformer::Operation *finalLayernorm = operations_.at(finalLayerNormId).get();
    std::vector<torch::Tensor> finalLayernormAtInTensors(finalLayernorm->GetInTensorCount());

    const int finalLayerNormWeightTensorId = weightTensors_.size() - 2;
    const int finalLayerNormBiasTensorId = weightTensors_.size() - 1;

    size_t layerNormInTensorNum = 0;
    finalLayernormAtInTensors.at(layerNormInTensorNum++) = firstInTensor;
    finalLayernormAtInTensors.at(layerNormInTensorNum++) = weightTensors_.at(finalLayerNormWeightTensorId);
    finalLayernormAtInTensors.at(layerNormInTensorNum++) = weightTensors_.at(finalLayerNormBiasTensorId);

    ExecuteOperation(finalLayerNormId, finalLayernormAtInTensors, outTensor, newOut);

    WaitAsyncPlanExecuteFinish();

    AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime += timer_.ElapsedMicroSecond();
    ASD_LOG(FATAL) << "ChatGlm6BModel executeCount:" << executeCount_ << ", statistic:["
                   << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString() << "]";
    AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();

    executeCount_++;
}

void ChatGlm6BModelDecoderAllTorch::BuildVariantPack(int opId, std::vector<torch::Tensor> &atInTensors,
                                                     torch::Tensor &outTensor, bool newOut,
                                                     AclTransformer::VariantPack &variantPack)
{
    variantPack.inTensors.clear();
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << opId << " atInTensors[" << i
                      << "].options:" << atInTensors.at(i).options() << ", data:" << atInTensors.at(i).data_ptr()
                      << ", storage_offset:" << atInTensors.at(i).storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atInTensors.at(i));
        variantPack.inTensors.push_back(Utils::AtTensor2AsdTensor(atInTensors.at(i)));
    }

    if (newOut) {
        if (!outTensors_.at(opId).numel()) {
            AclTransformer::Operation *operation = operations_.at(opId).get();
            AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
            operation->InferShape(variantPack.inTensors, outTensorDescs);
            outTensors_.at(opId) = Utils::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(0));
        }
        outTensor = outTensors_.at(opId);
    }
    variantPack.outTensors.clear();
    variantPack.outTensors.push_back(Utils::AtTensor2AsdTensor(outTensor));
}

void ChatGlm6BModelDecoderAllTorch::ExecuteOperation(int opId, std::vector<torch::Tensor> &opAtInTensors,
                                                     torch::Tensor &outTensor, bool newOut)
{
    AclTransformer::Plan &plan = *plans_.at(opId);

    AclTransformer::VariantPack &variantPack = variantPacks_.at(opId);
    variantPackParam_.layerId = opId - OPERATION_COUNT_BEFORE_LAYER;
    variantPack.param = variantPackParam_;
    BuildVariantPack(opId, opAtInTensors, outTensor, newOut, variantPack);

    AsdOps::Timer timer1;
    AsdOps::Status st = plan.Setup(handle_, variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime += timer1.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "ChatGlm6BModelLayer_" << opId << " setup plan fail, not call execute";
        return;
    }

    variantPack.workspaceSize = plan.GetWorkspaceSize();
    ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << opId << " get plan workspace size:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        variantPack.workspace =
            AsdOps::GetSingleton<AclTransformer::Context>().GetWorkspaceBuffer(variantPack.workspaceSize);
    }

    if (isUsePlanExecuteAsync_) {
        if (isTaskQueueEnable_) {
#ifdef TORCH_SETCUSTOMHANDLER
            at_npu::native::OpCommand cmd;
            cmd.Name("ChatGlm6BModelLayer_" + std::to_string(opId));
            cmd.SetCustomHandler([=]() {
                ExecutePlan(opId);
                return 0;
            });
            cmd.Run();
#else
            ASD_LOG(FATAL) << "ChatGlm6BModelLayer_" << opId << " torch_npu is low, can't support SetCustomHandler";
#endif
        } else {
            PushTask(opId);
        }
    } else {
        ExecutePlan(opId);
    }
}

void ChatGlm6BModelDecoderAllTorch::ParseParam(const std::string &param)
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

void ChatGlm6BModelDecoderAllTorch::ThreadProcessTask()
{
    ASD_LOG(FATAL) << "ChatGlm6BModelLayer ThreadProcessTask start";
    int ret = AsdRtDeviceSetCurrent(currentDevId_);
    ASD_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;

    size_t processTaskCount = 0;
    while (true) {
        int opId = PopTask();
        ExecutePlan(opId);
        processTaskCount++;
        if (processTaskCount == operations_.size()) {
            ASD_LOG(INFO) << "ChatGlm6BModelLayer thread process all layers";
            processTaskCount = 0;
            allTaskFinish_ = true;
        }
    }
}

void ChatGlm6BModelDecoderAllTorch::ExecutePlan(int opId)
{
    AsdOps::Timer timer2;
    AclTransformer::Plan &plan = *plans_.at(opId);
    AclTransformer::VariantPack &variantPack = variantPacks_.at(opId);
    ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << opId << " execute plan start";
    AsdOps::Status st = plan.Execute(handle_, variantPack);
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
        std::string dir = GetSaveTensorDir() + "/" + std::to_string(opId) + "_";
        plan.SetRunnerSaveTensorDir(dir);
    }

    AsdOps::GetSingleton<AclTransformer::Statistic>().planExecuteTime += timer2.ElapsedMicroSecond();
    ASD_LOG_IF(!st.Ok(), ERROR) << "ChatGlm6BModelOp_" << opId << " execute plan fail, error:" << st.Message();
}

void ChatGlm6BModelDecoderAllTorch::PushTask(int opId)
{
    std::unique_lock<std::mutex> lock(mutex_);
    taskQueue_.push(opId);
    lock.unlock();
    cond_.notify_one();
}

int ChatGlm6BModelDecoderAllTorch::PopTask()
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (taskQueue_.empty()) {
        cond_.wait(lock);
    }
    int opId = taskQueue_.front();
    taskQueue_.pop();
    return opId;
}

void ChatGlm6BModelDecoderAllTorch::WaitAsyncPlanExecuteFinish()
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

std::string ChatGlm6BModelDecoderAllTorch::GetSaveTensorDir()
{
    std::string dir = std::to_string(executeCount_) + "/0_ChatGlm6BModelDecoderAllTorch";
    return AclTransformer::Config::GetSaveTensorDir() + "/" + dir;
}

TORCH_LIBRARY(ChatGlm6BModelDecoderAllTorch, m)
{
    m.class_<ChatGlm6BModelDecoderAllTorch>("ChatGlm6BModelDecoderAllTorch")
        .def(torch::init<>())
        .def("set_param", &ChatGlm6BModelDecoderAllTorch::SetParam)
        .def("set_weight", &ChatGlm6BModelDecoderAllTorch::SetWeight)
        .def("execute", &ChatGlm6BModelDecoderAllTorch::Execute)
        .def("execute_out", &ChatGlm6BModelDecoderAllTorch::ExecuteOut);
}