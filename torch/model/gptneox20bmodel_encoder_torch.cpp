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
#include "gptneox20bmodel_encoder_torch.h"
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
#include "torch/utils/utils.h"
#include "torch/context/context.h"
#include "models/gptneox20b/gptneox20blayer_encoder_operation.h"

const size_t WEIGHT_COUNT_PER_LAYER = 12;

GptNeox20BModelEncoderTorch::GptNeox20BModelEncoderTorch()
{
    ASD_LOG(INFO) << "GptNeox20BModelEncoderTorch::GptNeox20BModelEncoderTorch, TASK_QUEUE_ENABLE:"
                  << c10_npu::option::OptionsManager().CheckQueueEnable();
    std::vector<AsdOps::Operation *> ops;
    AsdOps::Ops::Instance().GetAllOperations(ops);
}

GptNeox20BModelEncoderTorch::~GptNeox20BModelEncoderTorch() {}

void GptNeox20BModelEncoderTorch::SetParam(std::string param)
{
    ASD_LOG(INFO) << "GptNeox20BModel set param start, param:" << param;
    modelParam_.FromString(param);
    operations_.resize(modelParam_.layerNum);
    plans_.resize(modelParam_.layerNum);

    for (int i = 0; i < modelParam_.layerNum; ++i) {
        AclTransformer::GptNeox20BLayerParam opParam;
        opParam.layerNormEps = modelParam_.layerNormEps;
        opParam.headNum = modelParam_.headNum;
        opParam.transKey = modelParam_.transKey;
        opParam.dk = modelParam_.dk;
        opParam.layerId = i;
        opParam.rotaryPct = modelParam_.rotaryPct;
        AclTransformer::Operation *op = new AclTransformer::GptNeox20BLayerEncoderOperation(opParam);
        operations_.at(i).reset(op);
        AclTransformer::Plan *plan = new AclTransformer::Plan();
        op->BuildPlan(plan);
        plans_.at(i).reset(plan);
    }

    ASD_LOG(INFO) << "GptNeox20BModelEncoder set param end";
}

void GptNeox20BModelEncoderTorch::SetWeight(std::vector<torch::Tensor> weightTensors)
{
    if (weightTensors.size() != modelParam_.layerNum * WEIGHT_COUNT_PER_LAYER) {
        ASD_LOG(ERROR) << "GptNeox20BModelEncoder set weight fail, weightTensors.size:" << weightTensors.size()
                       << " != " << modelParam_.layerNum * WEIGHT_COUNT_PER_LAYER;
        return;
    }

    ASD_LOG(INFO) << "GptNeox20BModelEncoder set weight success, size:" << weightTensors.size();

    weightTensors_ = weightTensors;
    Utils::ContiguousAtTensor(weightTensors_);
}

std::vector<torch::Tensor> GptNeox20BModelEncoderTorch::Execute(torch::Tensor hiddenStateTensor,
                                                               torch::Tensor positionIdTensor,
                                                               torch::Tensor cosTableTensor,
                                                               torch::Tensor sinTableTensor,
                                                               torch::Tensor attentionMaskTensor, torch::Tensor seqLen)
{
    timer_.Reset();
    torch::Tensor outTensor;
    std::vector<torch::Tensor> presentKeyTensors(modelParam_.layerNum);
    std::vector<torch::Tensor> presentValueTensors(modelParam_.layerNum);

    ExecuteOutImpl(hiddenStateTensor, positionIdTensor, cosTableTensor, sinTableTensor, attentionMaskTensor, seqLen,
                   outTensor, presentKeyTensors, presentValueTensors, true);

    std::vector<torch::Tensor> outTensors(1 + modelParam_.layerNum * 2);
    size_t tensorId = 0;
    outTensors.at(tensorId++) = outTensor;
    for (int i = 0; i < modelParam_.layerNum; ++i) {
        outTensors.at(tensorId++) = presentKeyTensors.at(i);
    }
    for (int i = 0; i < modelParam_.layerNum; ++i) {
        outTensors.at(tensorId++) = presentValueTensors.at(i);
    }

    return outTensors;
}

void GptNeox20BModelEncoderTorch::ExecuteOut(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor,
                                            torch::Tensor cosTableTensor, torch::Tensor sinTableTensor,
                                            torch::Tensor attentionMaskTensor, torch::Tensor seqLen,
                                            torch::Tensor outTensor, std::vector<torch::Tensor> presentKeyTensors,
                                            std::vector<torch::Tensor> presentValueTensors)
{
    timer_.Reset();
    ExecuteOutImpl(hiddenStateTensor, positionIdTensor, cosTableTensor, sinTableTensor, attentionMaskTensor, seqLen,
                   outTensor, presentKeyTensors, presentValueTensors, false);
}

void GptNeox20BModelEncoderTorch::ExecuteOutImpl(torch::Tensor &hiddenStateTensor, torch::Tensor &positionIdTensor,
                                                torch::Tensor &cosTableTensor, torch::Tensor &sinTableTensor,
                                                torch::Tensor &attentionMaskTensor, torch::Tensor &seqLen,
                                                torch::Tensor &outTensor, std::vector<torch::Tensor> &presentKeyTensors,
                                                std::vector<torch::Tensor> &presentValueTensors, bool newOut)
{
    if ((int)presentKeyTensors.size() != modelParam_.layerNum ||
        (int)presentValueTensors.size() != modelParam_.layerNum) {
        ASD_LOG(ERROR) << "GptNeox20BModelEncoder presentKeyTensors.size:" << presentKeyTensors.size()
                       << ", presentValueTensors.size:" << presentValueTensors.size();
        return;
    }

    handle_ = {Utils::GetCurrentStream()};

    Utils::ContiguousAtTensor(hiddenStateTensor);
    Utils::ContiguousAtTensor(positionIdTensor);
    Utils::ContiguousAtTensor(cosTableTensor);
    Utils::ContiguousAtTensor(sinTableTensor);
    Utils::ContiguousAtTensor(attentionMaskTensor);
//    Utils::ContiguousAtTensor(seqLen);
    if (!newOut) {
        Utils::ContiguousAtTensor(outTensor);
        Utils::ContiguousAtTensor(presentKeyTensors);
        Utils::ContiguousAtTensor(presentValueTensors);
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
//        opAtInTensors.at(inTensorId++) = seqLen;              // seqLen

        ExecuteSingleOperation(layerId, opAtInTensors, outTensor, presentKeyTensors.at(layerId),
                               presentValueTensors.at(layerId), newOut);

        firstInTensor = outTensor;
    }

    AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime += timer_.ElapsedMicroSecond();
    ASD_LOG(FATAL) << "GptNeox20BModelEncoder executeCount:" << executeCount_ << ", statistic:["
                   << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString() << "]";
    AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();

    executeCount_++;
}

void GptNeox20BModelEncoderTorch::BuildVariantPack(int layerId, std::vector<torch::Tensor> &atInTensors,
                                                  torch::Tensor &outTensor, torch::Tensor &presentKeyTensor,
                                                  torch::Tensor &presentValueTensor, bool newOut,
                                                  AclTransformer::VariantPack &variantPack)
{
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << "GptNeox20BModelEncoderLayer_" << layerId << " atInTensors[" << i
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
        presentKeyTensor = Utils::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(1));
        presentValueTensor = Utils::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(2));
    }

    variantPack.outTensors.push_back(Utils::AtTensor2AsdTensor(outTensor));
    variantPack.outTensors.push_back(Utils::AtTensor2AsdTensor(presentKeyTensor));
    variantPack.outTensors.push_back(Utils::AtTensor2AsdTensor(presentValueTensor));
}

void GptNeox20BModelEncoderTorch::ExecuteSingleOperation(int layerId, std::vector<torch::Tensor> &opAtInTensors,
                                                        torch::Tensor &outTensor, torch::Tensor &presentKeyTensor,
                                                        torch::Tensor &presentValueTensor, bool newOut)
{
    AclTransformer::Plan &plan = *plans_.at(layerId);

    AclTransformer::VariantPack variantPack;
    BuildVariantPack(layerId, opAtInTensors, outTensor, presentKeyTensor, presentValueTensor, newOut, variantPack);

    AsdOps::Timer timer1;
    AsdOps::Status st = plan.Setup(handle_, variantPack);
    AsdOps::GetSingleton<AclTransformer::Statistic>().planSetupTime += timer1.ElapsedMicroSecond();
    if (!st.Ok()) {
        ASD_LOG(ERROR) << "GptNeox20BModelEncoderLayer_" << layerId << " setup plan fail, not call execute";
        return;
    }

    variantPack.workspaceSize = plan.GetWorkspaceSize();
    ASD_LOG(INFO) << "GptNeox20BModelEncoderLayer_" << layerId << " get plan workspace size:" << variantPack.workspaceSize;

    if (variantPack.workspaceSize > 0) {
        variantPack.workspace =
            AsdOps::GetSingleton<AclTransformer::Context>().GetWorkspaceBuffer(variantPack.workspaceSize);
    }

    AsdOps::Timer timer2;
    st = plan.Execute(handle_, variantPack);

    if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
        AsdRtStreamSynchronize(handle_.stream);
        std::string dirPath =
            AclTransformer::Config::GetSaveTensorDir() + "/GptNeox20BModelEncoderLayer_" + std::to_string(layerId);
        AclTransformer::TensorUtil::SaveVariantPack(handle_, variantPack, dirPath);
        ASD_LOG(FATAL) << "GptNeox20BModelEncoderLayer_" << layerId << " save variant pack, dir:" << dirPath;
    }

    AsdOps::GetSingleton<AclTransformer::Statistic>().planExecuteTime += timer2.ElapsedMicroSecond();
    ASD_LOG_IF(!st.Ok(), ERROR) << "GptNeox20BModelEncoderLayer_" << layerId << " execute plan fail, error:" << st.Message();
}

TORCH_LIBRARY(GptNeox20BModelEncoderTorch, m)
{
    m.class_<GptNeox20BModelEncoderTorch>("GptNeox20BModelEncoderTorch")
        .def(torch::init<>())
        .def("set_param", &GptNeox20BModelEncoderTorch::SetParam)
        .def("set_weight", &GptNeox20BModelEncoderTorch::SetWeight)
        .def("execute", &GptNeox20BModelEncoderTorch::Execute)
        .def("execute_out", &GptNeox20BModelEncoderTorch::ExecuteOut);
}