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
#include "model_torch.h"
#include <asdops/ops.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/statistic.h"
#include "acltransformer/config.h"
#include "model.h"
#include "torch/utils/utils.h"
#include "torch/model_v2/chatglm6b/chatglm6b_decoder_model.h"
#include "torch/model_v2/chatglm2_6b/chatglm2_6b_decoder_model.h"
#include "torch/model_v2/glm130b/glm130b_decoder_model.h"
#include "torch/model_v2/glm130b/glm130b_decoder_fusion_model.h"
#include "torch/model_v2/chatglm2_6b/chatglm2_6b_encoder_model.h"
#include "torch/model_v2/chatglm6b/chatglm6b_decoder_without_fusion_model.h"
#include "torch/model_v2/chatglm6b/chatglm6b_encoder_without_fusion_model.h"
#include "torch/model_v2/chatglm6b/chatglm6bmodel_decoder_quant_flash_model.h"
#include "torch/model_v2/chatglm2_6b/chatglm2_6b_decoder_flashattention_model.h"
#include "torch/model_v2/glm130b/glm130b_decoder_model_post_operation.h"
#include "torch/model_v2/gptneox20b/gptneox20b_decoder_model.h"
#include "torch/model_v2/gptneox20b/gptneox20b_encoder_model.h"
#include "torch/model_v2/bloom7b/bloom7b_decoder_model.h"
#include "torch/model_v2/chatglm6b/chatglm6b_encoder_quant_model.h"
#include "torch/model_v2/chatglm6b/chatglm6b_decoder_quant_model.h"
#include "torch/model_v2/baichuan1_7b/baichuan1_7b_decoder_model.h"
#include "torch/model_v2/baichuan1_7b/baichuan1_7b_encoder_model.h"
#include "torch/model_v2/baichuan1_7b/baichuan1_7b_encoder_with_bias_model.h"
#include "torch/model_v2/baichuan2_7b/baichuan2_7b_decoder_model.h"

uint64_t GetNewModelId()
{
    static uint64_t modelId = 0;
    uint64_t newModelId = modelId++;
    return newModelId;
}

ModelTorch::ModelTorch(std::string modelName) : modelName_(modelName)
{
    std::vector<AsdOps::Operation *> ops;
    AsdOps::Ops::Instance().GetAllOperations(ops);
    modelId_ = GetNewModelId();
    ASD_LOG(INFO) << "ModelTorch new modelName:" << modelName_ << ", modelId:" << modelId_;
}

ModelTorch::~ModelTorch() {}

void ModelTorch::SetParam(std::string param)
{
    ASD_LOG(INFO) << "ModelTorch set param start, modelName:" << modelName_ << ", param:" << param;
    if (modelName_ == "ChatGlm6BDecoderModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm6BDecoderModel>(param);
    } else if (modelName_ == "ChatGlm2DecoderModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm2DecoderModel>(param);
    } else if (modelName_ == "Glm130BDecoderModel") {
        model_ = std::make_shared<AclTransformer::Glm130BDecoderModel>(param);
    } else if (modelName_ == "Glm130BDecoderModelWithFusion") {
        model_ = std::make_shared<AclTransformer::Glm130BDecoderFusionModel>(param);
    } else if (modelName_ == "ChatGlm2EncoderModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm2EncoderModel>(param);
    } else if (modelName_ == "ChatGlm6BDecoderWithoutFusionModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm6BDecoderWithoutFusionModel>(param);
    } else if (modelName_ == "ChatGlm6BEncoderWithoutFusionModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm6BEncoderWithoutFusionModel>(param);
    } else if (modelName_ == "ChatGlm6BDecoderQuantFlashModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm6BDecoderQuantFlashModel>(param);
    } else if (modelName_ == "ChatGlm2DecoderFlashAttentionModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm2DecoderFlashAttentionModel>(param);
    } else if (modelName_ == "Glm130BDecoderPostOperationModel") {
        model_ = std::make_shared<AclTransformer::Glm130BDecoderPostOperationModel>(param);
    } else if (modelName_ == "GptNeox20BDecoderModel") {
        model_ = std::make_shared<AclTransformer::GptNeox20BDecoderModel>(param);
    } else if (modelName_ == "GptNeox20BEncoderModel") {
        model_ = std::make_shared<AclTransformer::GptNeox20BEncoderModel>(param);
    } else if (modelName_ == "Bloom7BDecoderModel") {
        model_ = std::make_shared<AclTransformer::Bloom7BDecoderModel>(param);
    } else if (modelName_ == "ChatGlm6BEncoderQuantModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm6BEncoderQuantModel>(param);
    } else if (modelName_ == "ChatGlm6BDecoderQuantModel") {
        model_ = std::make_shared<AclTransformer::ChatGlm6BDecoderQuantModel>(param);
    } else if (modelName_ == "BaiChuan17BDecoderModel") {
        model_ = std::make_shared<AclTransformer::BaiChuan17BDecoderModel>(param);
    } else if (modelName_ == "BaiChuan27BDecoderModel") {
        model_ = std::make_shared<AclTransformer::BaiChuan27BDecoderModel>(param);
    } else if (modelName_ == "BaiChuan17BEncoderModel") {
        model_ = std::make_shared<AclTransformer::BaiChuan17BEncoderModel>(param);
    } else if (modelName_ == "BaiChuan17BEncoderWithBiasModel") {
        model_ = std::make_shared<AclTransformer::BaiChuan17BEncoderWithBiasModel>(param);
    } else {
        ASD_LOG(FATAL) << "not support modelName:" << modelName_;
        return;
    }

    model_->Init();

    ASD_LOG(INFO) << "ModelTorch set param end";
}

void ModelTorch::SetWeight(std::vector<torch::Tensor> atWeightTensors)
{
    ASD_LOG(INFO) << "ModelTorch set weight:" << atWeightTensors.size();
    for (size_t i = 0; i < atWeightTensors.size(); ++i) {
        const torch::Tensor &atTensor = atWeightTensors.at(i);
        ASD_LOG(INFO) << "ModelTorch atWeightTensors[" << i << "]"
                      << " data:" << atTensor.data_ptr() << ", storage_offset:" << atTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atTensor) << ", shape:" << atTensor.sizes()
                      << ", options:" << atTensor.options();
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = GetSaveTensorDir() + "/weight" + std::to_string(i) + ".pth";
            Utils::SaveTensor(atTensor, filePath);
            ASD_LOG(INFO) << "ModelTorch save weight tensor:" << filePath;
        }
    }
    std::vector<AsdOps::Tensor> weigthTensors;
    AtTensor2AsdTensor(atWeightTensors, weigthTensors);
    model_->SetWeight(weigthTensors);
}

std::vector<torch::Tensor> ModelTorch::Execute(std::vector<torch::Tensor> atInTensors, std::string param)
{
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        const torch::Tensor &atTensor = atInTensors.at(i);
        ASD_LOG(INFO) << "ModelTorch atInTensors[" << i << "]"
                      << " data:" << atTensor.data_ptr() << ", storage_offset:" << atTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atTensor) << ", shape:" << atTensor.sizes()
                      << ", options:" << atTensor.options();
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = GetSaveTensorDir() + "/intensor" + std::to_string(i) + ".pth";
            Utils::SaveTensor(atTensor, filePath);
            ASD_LOG(INFO) << "ModelTorch save weight tensor:" << filePath;
        }
    }

    std::vector<AsdOps::Tensor> inTensors;
    AtTensor2AsdTensor(atInTensors, inTensors);
    if (AsdOps::GetSingleton<AclTransformer::Config>().IsConvertNCHWToND()) {
        for (size_t i = 0; i < inTensors.size(); ++i) {
            if (inTensors.at(i).desc.format == AsdOps::TENSOR_FORMAT_NCHW) {
                inTensors.at(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
            }
        }
    }

    std::vector<AsdOps::TensorDesc> outTensorDescs(model_->GetOutTensorCount());
    AsdOps::Status st = model_->InferShape(inTensors, outTensorDescs);
    ASD_LOG_IF(!st.Ok(), FATAL) << "ModelTorch infer shape fail, error:" << st.Message();

    std::vector<torch::Tensor> atOutTensors(outTensorDescs.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ASD_LOG(INFO) << "ModelTorch outTensorDescs[" << i
                      << "]:" << AclTransformer::TensorUtil::AsdOpsTensorDescToString(outTensorDescs.at(i));
        AsdOps::Timer timer;
        atOutTensors.at(i) = Utils::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(i));
        AsdOps::GetSingleton<AclTransformer::Statistic>().createTensorTime += timer.ElapsedMicroSecond();
    }

    std::vector<AsdOps::Tensor> outTensors;
    AtTensor2AsdTensor(atOutTensors, outTensors);

    ExecuteOutImpl(inTensors, outTensors, param);

    return atOutTensors;
}

void ModelTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                            std::string param)
{
    std::vector<AsdOps::Tensor> inTensors;
    AtTensor2AsdTensor(atInTensors, inTensors);

    std::vector<AsdOps::Tensor> outTensors;
    AtTensor2AsdTensor(atOutTensors, outTensors);

    ExecuteOutImpl(inTensors, outTensors, param);
}

void ModelTorch::ExecuteOutImpl(std::vector<AsdOps::Tensor> &inTensors, std::vector<AsdOps::Tensor> &outTensors,
                                const std::string &param)
{
    AclTransformer::Handle handle = {Utils::GetCurrentStream()};
    model_->Execute(handle, inTensors, outTensors, param);
    executeCount_++;
}

void ModelTorch::AtTensor2AsdTensor(std::vector<torch::Tensor> &atTensors, std::vector<AsdOps::Tensor> &opsTensors)
{
    for (auto &atTensor : atTensors) {
        Utils::ContiguousAtTensor(atTensor);
        AsdOps::Tensor tensor = Utils::AtTensor2AsdTensor(atTensor);
        opsTensors.push_back(tensor);
    }
}

std::string ModelTorch::GetSaveTensorDir()
{
    std::string dir = std::to_string(executeCount_) + "/" + std::to_string(modelId_) + "_ModelTorch";
    return AclTransformer::Config::GetSaveTensorDir() + "/" + dir;
}

TORCH_LIBRARY(ModelTorch, m)
{
    m.class_<ModelTorch>("ModelTorch")
        .def(torch::init<std::string>())
        .def("set_param", &ModelTorch::SetParam)
        .def("set_weight", &ModelTorch::SetWeight)
        .def("execute", &ModelTorch::Execute)
        .def("execute_out", &ModelTorch::ExecuteOut);
}