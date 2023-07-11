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
#include "chatglm6bmodel_rope_torch.h"
#include <nlohmann/json.hpp>
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
#include "examples/ops/chatglm6b/chatglm6blayer_rope_operation.h"

const size_t WEIGHT_COUNT_PER_LAYER = 12;

void ChatGlm6BModelRopeParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerNormEps = paramJson["layerNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    transKey = paramJson["transKey"].get<bool>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    residualAddScale = paramJson["residualAddScale"].get<float>();
}

ChatGlm6BModelRopeTorch::ChatGlm6BModelRopeTorch()
{
    ASD_LOG(INFO) << "ChatGlm6BModelRopeTorch::ChatGlm6BModelRopeTorch, TASK_QUEUE_ENABLE:"
                  << c10_npu::option::OptionsManager().CheckQueueEnable();
    std::vector<AsdOps::Operation *> ops;
    AsdOps::Ops::Instance().GetAllOperations(ops);
}

ChatGlm6BModelRopeTorch::~ChatGlm6BModelRopeTorch() {}

void ChatGlm6BModelRopeTorch::SetParam(std::string param)
{
    ASD_LOG(INFO) << "ChatGlm6BModel set param start, param:" << param;
    modelParam_.FromString(param);
    operations_.resize(modelParam_.layerNum);
    planv2s_.resize(modelParam_.layerNum);

    for (int i = 0; i < modelParam_.layerNum; ++i) {
        AclTransformer::ChatGlm6BLayerParam opParam;
        opParam.layerNormEps = modelParam_.layerNormEps;
        opParam.headNum = modelParam_.headNum;
        opParam.transKey = modelParam_.transKey;
        opParam.dk = modelParam_.dk;
        opParam.layerId = i;
        opParam.residualAddScale = modelParam_.residualAddScale;
        AclTransformer::Operation *op = new AclTransformer::ChatGlm6BLayerRopeOperation(opParam);
        operations_.at(i).reset(op);
        AclTransformer::PlanV2 *planv2 = new AclTransformer::PlanV2();
        op->BuildPlan(planv2);
        planv2s_.at(i).reset(planv2);
    }

    ASD_LOG(INFO) << "ChatGlm6BModel set param end";
}

void ChatGlm6BModelRopeTorch::SetWeight(std::vector<torch::Tensor> weightTensors)
{
    if (weightTensors.size() != modelParam_.layerNum * WEIGHT_COUNT_PER_LAYER) {
        ASD_LOG(ERROR) << "ChatGlm6BModel set weight fail, weightTensors.size:" << weightTensors.size()
                       << " != " << modelParam_.layerNum * WEIGHT_COUNT_PER_LAYER;
        return;
    }

    ASD_LOG(INFO) << "ChatGlm6BModel set weight success, size:" << weightTensors.size();

    weightTensors_ = weightTensors;
    ExampleUtil::ContiguousAtTensor(weightTensors_);
}

std::vector<torch::Tensor> ChatGlm6BModelRopeTorch::Execute(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor,
                                                        torch::Tensor cosTableTensor, torch::Tensor sinTableTensor,
                                                        torch::Tensor attentionMaskTensor,
                                                        std::vector<torch::Tensor> pastKeyTensors,
                                                        std::vector<torch::Tensor> pastValueTensors,
                                                        torch::Tensor seqLen)
{
    timer_.Reset();
    torch::Tensor outTensor;
    std::vector<torch::Tensor> presendKeyTensors(modelParam_.layerNum);
    std::vector<torch::Tensor> presentValueTensors(modelParam_.layerNum);

    ExecuteOutImpl(hiddenStateTensor, positionIdTensor, cosTableTensor, sinTableTensor, attentionMaskTensor,
                   pastKeyTensors, pastValueTensors, outTensor, presendKeyTensors, presentValueTensors, seqLen, true);

    std::vector<torch::Tensor> outTensors(1 + modelParam_.layerNum * 2);
    size_t tensorId = 0;
    outTensors.at(tensorId++) = outTensor;
    for (int i = 0; i < modelParam_.layerNum; ++i) {
        outTensors.at(tensorId++) = presendKeyTensors.at(i);
    }
    for (int i = 0; i < modelParam_.layerNum; ++i) {
        outTensors.at(tensorId++) = presentValueTensors.at(i);
    }

    return outTensors;
}

void ChatGlm6BModelRopeTorch::ExecuteOut(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor,
                                     torch::Tensor cosTableTensor, torch::Tensor sinTableTensor,
                                     torch::Tensor attentionMaskTensor, std::vector<torch::Tensor> pastKeyTensors,
                                     std::vector<torch::Tensor> pastValueTensors, torch::Tensor outTensor,
                                     std::vector<torch::Tensor> presendKeyTensors,
                                     std::vector<torch::Tensor> presentValueTensors, torch::Tensor seqLen)
{
    timer_.Reset();
    ExecuteOutImpl(hiddenStateTensor, positionIdTensor, cosTableTensor, sinTableTensor, attentionMaskTensor,
                   pastKeyTensors, pastValueTensors, outTensor, presendKeyTensors, presentValueTensors, seqLen, false);
}

void ChatGlm6BModelRopeTorch::ExecuteOutImpl(torch::Tensor &hiddenStateTensor, torch::Tensor &positionIdTensor,
                                         torch::Tensor &cosTableTensor, torch::Tensor &sinTableTensor,
                                         torch::Tensor &attentionMaskTensor, std::vector<torch::Tensor> &pastKeyTensors,
                                         std::vector<torch::Tensor> &pastValueTensors, torch::Tensor &outTensor,
                                         std::vector<torch::Tensor> &presendKeyTensors,
                                         std::vector<torch::Tensor> &presentValueTensors, torch::Tensor &seqLen,
                                         bool newOut)
{
    if ((int)pastKeyTensors.size() != modelParam_.layerNum || (int)pastValueTensors.size() != modelParam_.layerNum ||
        (int)presendKeyTensors.size() != modelParam_.layerNum ||
        (int)presentValueTensors.size() != modelParam_.layerNum) {
        ASD_LOG(ERROR) << "ChatGlm6BModel pastKeyTensors.size:" << pastKeyTensors.size()
                       << ", pastValueTensors.size:" << pastValueTensors.size()
                       << ", presendKeyTensors.size:" << presendKeyTensors.size()
                       << ", presentValueTensors.size:" << presentValueTensors.size();
        return;
    }

    handle_ = {ExampleUtil::GetCurrentStream()};

    ExampleUtil::ContiguousAtTensor(hiddenStateTensor);
    ExampleUtil::ContiguousAtTensor(positionIdTensor);
    ExampleUtil::ContiguousAtTensor(cosTableTensor);
    ExampleUtil::ContiguousAtTensor(sinTableTensor);
    ExampleUtil::ContiguousAtTensor(attentionMaskTensor);
    ExampleUtil::ContiguousAtTensor(pastKeyTensors);
    ExampleUtil::ContiguousAtTensor(pastValueTensors);
    ExampleUtil::ContiguousAtTensor(seqLen);
    if (!newOut) {
        ExampleUtil::ContiguousAtTensor(outTensor);
        ExampleUtil::ContiguousAtTensor(presendKeyTensors);
        ExampleUtil::ContiguousAtTensor(presentValueTensors);
    }

    AclTransformer::Operation *operation = operations_.at(0).get();
    std::vector<torch::Tensor> opAtInTensors(operation->GetInTensorCount());
    std::vector<torch::Tensor> opAtOutTensors(operation->GetOutTensorCount());

    torch::Tensor firstInTensor = hiddenStateTensor;
    for (int layerId = 0; layerId < modelParam_.layerNum; ++layerId) {
        size_t inTensorId = 0;
        opAtInTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            opAtInTensors.at(inTensorId++) = weightTensors_.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        opAtInTensors.at(inTensorId++) = positionIdTensor;    // positionIdTensor
        opAtInTensors.at(inTensorId++) = cosTableTensor;      // cosTable
        opAtInTensors.at(inTensorId++) = sinTableTensor;      // sinTable
        opAtInTensors.at(inTensorId++) = attentionMaskTensor; // attentionMaskTensor
        opAtInTensors.at(inTensorId++) = pastKeyTensors.at(layerId);
        opAtInTensors.at(inTensorId++) = pastValueTensors.at(layerId);
        opAtInTensors.at(inTensorId++) = seqLen;              // seqLen

        ExecuteSingleOperation(layerId, opAtInTensors, outTensor, presendKeyTensors.at(layerId),
                               presentValueTensors.at(layerId), newOut);

        firstInTensor = outTensor;
    }

    AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime += timer_.ElapsedMicroSecond();
    ASD_LOG(FATAL) << "ChatGlm6BModel executeCount:" << executeCount_ << ", statistic:["
                   << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString() << "]";
    AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();

    executeCount_++;
}

void ChatGlm6BModelRopeTorch::BuildVariantPack(int layerId, std::vector<torch::Tensor> &atInTensors,
                                           torch::Tensor &outTensor, torch::Tensor &presendKeyTensor,
                                           torch::Tensor &presentValueTensor, bool newOut,
                                           AclTransformer::VariantPack &variantPack)
{
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << layerId << " atInTensors[" << i
                      << "].options:" << atInTensors.at(i).options() << ", data:" << atInTensors.at(i).data_ptr()
                      << ", storage_offset:" << atInTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atInTensors.at(i));
        variantPack.inTensors.push_back(ExampleUtil::AtTensor2AsdTensor(atInTensors.at(i)));
    }

    if (newOut) {
        AclTransformer::Operation *operation = operations_.at(layerId).get();
        AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
        operation->InferShape(variantPack.inTensors, outTensorDescs);
        outTensor = ExampleUtil::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(0));
        presendKeyTensor = ExampleUtil::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(1));
        presentValueTensor = ExampleUtil::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(2));
    }

    variantPack.outTensors.push_back(ExampleUtil::AtTensor2AsdTensor(outTensor));
    variantPack.outTensors.push_back(ExampleUtil::AtTensor2AsdTensor(presendKeyTensor));
    variantPack.outTensors.push_back(ExampleUtil::AtTensor2AsdTensor(presentValueTensor));
}

void ChatGlm6BModelRopeTorch::ExecuteSingleOperation(int layerId, std::vector<torch::Tensor> &opAtInTensors,
                                                 torch::Tensor &outTensor, torch::Tensor &presendKeyTensor,
                                                 torch::Tensor &presentValueTensor, bool newOut)
{
    AclTransformer::PlanV2 &planv2 = *planv2s_.at(layerId);

    AclTransformer::VariantPack variantPack;
    BuildVariantPack(layerId, opAtInTensors, outTensor, presendKeyTensor, presentValueTensor, newOut, variantPack);

    AsdOps::Timer timer1;
    AsdOps::Status st = planv2.Setup(handle_, variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime += timer1.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "ChatGlm6BModelLayer_" << layerId << " setup plan fail, not call execute";
        return;
    }

    variantPack.workspaceSize = planv2.GetWorkspaceSize();
    ASD_LOG(INFO) << "ChatGlm6BModelLayer_" << layerId << " get plan workspace size:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        AsdOps::GetSingleton<AclTransformer::Workspace>().SetWorkspace(variantPack.workspaceSize);
        variantPack.workspace = AsdOps::GetSingleton<AclTransformer::Workspace>().GetWorkspace();
    }

    AsdOps::Timer timer2;
    st = planv2.Execute(handle_, variantPack);

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

TORCH_LIBRARY(ChatGlm6BModelRopeTorch, m)
{
    m.class_<ChatGlm6BModelRopeTorch>("ChatGlm6BModelRopeTorch")
        .def(torch::init<>())
        .def("set_param", &ChatGlm6BModelRopeTorch::SetParam)
        .def("set_weight", &ChatGlm6BModelRopeTorch::SetWeight)
        .def("execute", &ChatGlm6BModelRopeTorch::Execute)
        .def("execute_out", &ChatGlm6BModelRopeTorch::ExecuteOut);
}