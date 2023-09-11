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
#include "operation_creator.h"
#include <nlohmann/json.hpp>
#include <functional>
#include <atb/atb_infer.h>
#include "chatglm6b/layer/chatglm6blayer_encoder_operation.h"
#include "chatglm6b/layer/chatglm6blayer_decoder_flashattention_operation.h"
#include "chatglm6b/layer/chatglm6blayer_decoder_without_fusion_operation.h"
#include "chatglm6b/operation/chatglm6b_lmhead_operation.h"
#include "llama_parallel/layer/llamalayer_encoder_parallel_operation.h"
#include "llama_parallel/layer/llamalayer_fusion_parallel_operation.h"

using OperationCreateFunc = std::function<atb::Operation *(const nlohmann::json &paramJson)>;

static atb::Operation *AllReduceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllReduceParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("allReduceType") != paramJson.end()) {
        param.allReduceType = paramJson["allReduceType"].get<std::string>();
    }
    ATB_LOG(INFO) << "AllReduceParam rank:" << param.rank;
    ATB_LOG(INFO) << "AllReduceParam rankSize:" << param.rankSize;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *AllGatherOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllGatherParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    ATB_LOG(INFO) << "AllGatherParam rank:" << param.rank;
    ATB_LOG(INFO) << "AllGatherParam rankSize:" << param.rankSize;
    ATB_LOG(INFO) << "AllGatherParam backend:" << param.backend;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *LinearParallelOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::LinearParallelParam param;
    if (paramJson.find("transWeight") != paramJson.end()) {
        param.transWeight = paramJson["transWeight"].get<bool>();
    }
    if (paramJson.find("bias") != paramJson.end()) {
        param.bias = paramJson["bias"].get<std::string>();
    }
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.parallelType = paramJson["parallelType"].get<std::string>();
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *RopeOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::PositionEmbeddingFusionParam param;
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<std::int64_t>();
        ATB_LOG(INFO) << "param.headNum: " << param.headNum;
    }
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *PositionEmbedding1dSplitFusionOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::PositionEmbedding1dFusionParam param;
    param.headNum = paramJson["headNum"].get<int64_t>();
    ATB_LOG(INFO) << "param.headNum: " << param.headNum;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *AddNormOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::AddNormParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    ATB_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps;
    param.zoom_scale = paramJson["zoom_scale"].get<float>();
    ATB_LOG(INFO) << "NormParam zoom_scale:" << param.zoom_scale;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *RmsNormOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::RmsNormParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<double>();
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *EmbeddingOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::EmbeddingParam param;
    ATB_LOG(INFO) << "EmbeddingParam axis:" << param.axis;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *NormOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::NormParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    if (paramJson.contains("beginNormAxis")) {
        param.beginNormAxis = paramJson["beginNormAxis"].get<int32_t>();
    }
    if (paramJson.contains("beginParamsAxis")) {
        param.beginParamsAxis = paramJson["beginParamsAxis"].get<int32_t>();
    }
    ATB_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps << ", beginNormAxis:" << param.beginNormAxis
                  << ", beginParamsAxis:" << param.beginParamsAxis;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *LinearOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    ATB_LOG(INFO) << "LinearParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *FfnOldOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearActivationParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("activationFuncType")) {
        param.activationFuncType =
            atb::infer::ActivationType(paramJson["activationFuncType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "FfnParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias << ", activationFuncType:" << param.activationFuncType;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *FfnOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::FfnParam param;
    param.firstTransposeA = paramJson["firstTransposeA"].get<bool>();
    param.firstTransposeB = paramJson["firstTransposeB"].get<bool>();
    param.secondTransposeA = paramJson["secondTransposeA"].get<bool>();
    param.secondTransposeB = paramJson["secondTransposeB"].get<bool>();
    if (paramJson.contains("firstHasBias")) {
        param.firstHasBias = paramJson["firstHasBias"].get<bool>();
    }

    if (paramJson.contains("secondHasBias")) {
        param.secondHasBias = paramJson["secondHasBias"].get<bool>();
    }

    if (paramJson.contains("activationType")) {
        param.activationType =
            atb::infer::ActivationType(paramJson["activationType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "FfnParam firstTransposeA:" << param.firstTransposeA << ", firstTransposeB:" << param.firstTransposeB
                  << ", firstHasBias:" << param.firstHasBias << ", activationType:" << param.activationType 
                  << "FfnParam secondTransposeA:" << param.secondTransposeA << ", secondTransposeB:" << param.secondTransposeB
                  << ", secondHasBias:" << param.secondHasBias;

    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *MlpOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::MlpParam param;
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
        ATB_LOG(INFO) << "MlpParam model:" << param.model;
    } else {
        param.model = "llama7b";
        ATB_LOG(INFO) << "MlpParam is empty, default model:" << param.model;
    }
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *MlpQuantOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::MlpQuantParam param;
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
        ATB_LOG(INFO) << "MlpQuantParam model:" << param.model;
    } else {
        param.model = "llama7b";
        ATB_LOG(INFO) << "MlpQuantParam is empty, default model:" << param.model;
    }

    param.inputScale = paramJson["inputScale"].get<double>();
    param.inputOffset = paramJson["inputOffset"].get<int>();
    ATB_LOG(INFO) << "MlpQuantParams: "
                  << ", input_scale:" << param.inputScale << ", input_offset:" << param.inputOffset;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *SelfAttentionOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::SelfAttentionParam param;
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    if (paramJson.contains("preScale")) {
        param.preScale = paramJson["preScale"].get<float>();
    }
    if (paramJson.contains("postScale")) {
        param.postScale = paramJson["postScale"].get<float>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    ATB_LOG(INFO) << "SelfAttentionKvCacheParam transKey:" << param.transKey << ", headNum:" << param.headNum
                  << ", layerId:" << param.layerId << ", dk:" << param.dk << ", preScale" << param.preScale
                  << ", postScale" << param.postScale << ", model" << param.model << ", hiddenSizePerHead"
                  << param.hiddenSizePerHead;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *PositionEmbedding1dSplitOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::PositionEmbedding1dSplitParam param;
    param.headNum = paramJson["headNum"].get<int>();
    ATB_LOG(INFO) << "PositionEmbeddingParam headNum:" << param.headNum;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *PositionEmbeddingOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::PositionEmbeddingParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int64_t>();
        ATB_LOG(INFO) << "PositionEmbeddingParam headNum:" << param.headNum;
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    } else {
        param.model = "llama7b";
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("rotaryPct")) {
        param.rotaryPct = paramJson["rotaryPct"].get<float>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int64_t>();
    }
    if (paramJson.contains("is2d")) {
        param.is2d = paramJson["is2d"].get<bool>();
    }
    if (paramJson.contains("isFusion")) {
        param.isFusion = paramJson["isFusion"].get<bool>();
    }
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *SelfAttentionKvCacheOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::SelfAttentionKvCacheParam param;
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    if (paramJson.contains("preScale")) {
        param.preScale = paramJson["preScale"].get<float>();
    }
    if (paramJson.contains("postScale")) {
        param.postScale = paramJson["postScale"].get<float>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("invNormFactorvarAttr")) {
        param.invNormFactorvarAttr = paramJson["invNormFactorvarAttr"].get<float>();
    }
    ATB_LOG(INFO) << "SelfAttentionKvCacheParam transKey:" << param.transKey << ", headNum:" << param.headNum
                  << ", layerId:" << param.layerId << ", dk:" << param.dk << ", preScale" << param.preScale
                  << ", postScale" << param.postScale << ", model" << param.model << ", hiddenSizePerHead"
                  << param.hiddenSizePerHead << ", numHeadsPerPartition" << param.numHeadsPerPartition
                  << ", numGroupsPerPartition" << param.numGroupsPerPartition << ", invNormFactorvarAttr"
                  << param.invNormFactorvarAttr;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *TransposeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransposeParam param;
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "transpose(" << param.perm << ")";
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *FfnQuantOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::FfnQuantParam param;
    param.firstLinearParam.transposeA = paramJson["firstTransposeA"].get<bool>();
    param.firstLinearParam.transposeB = paramJson["firstTransposeB"].get<bool>();

    param.secondLinearParam.transposeA = paramJson["secondTransposeA"].get<bool>();
    param.secondLinearParam.transposeB = paramJson["secondTransposeB"].get<bool>();

    param.inputScale = paramJson["inputScale"].get<float>();
    param.inputOffset = paramJson["inputOffset"].get<int>();

    ATB_LOG(INFO) << "firstLinearParam transposeA:" << param.firstLinearParam.transposeA << ", transposeB:"<<
                  param.firstLinearParam.transposeB << "secondLinearParam transposeA:" << param.secondLinearParam.transposeA <<
                  ", transposeB:"<< param.secondLinearParam.transposeB << "input_scale" << param.inputScale <<
                  "input_offset" << param.inputOffset ;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *LinearActivationOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearActivationParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("activationFuncType")) {
        param.activationFuncType =
            atb::infer::ActivationType(paramJson["activationFuncType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "LinearActivationParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias << ", activationFuncType:" << param.activationFuncType;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *LinearQuantOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::LinearQuantParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    ATB_LOG(INFO) << "LinearQuantParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *AddNormQuantOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::AddNormQuantParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.inputScale = paramJson["input_scale"].get<float>();
    param.inputOffset = paramJson["input_offset"].get<int>();
    param.inputAlpha = paramJson["input_alpha"].get<float>();

    ATB_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps << ", input_scale:" << param.inputScale
                  << ", input_offset:" << param.inputOffset << ", input_alpha:" << param.inputAlpha;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *NormQuantOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::NormQuantParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.inputScale = paramJson["input_scale"].get<float>();
    param.inputOffset = paramJson["input_offset"].get<int>();
    param.inputAlpha = paramJson["input_alpha"].get<float>();

    ATB_LOG(INFO) << "NormParam layerNormEps:" << param.layerNormEps << ", input_scale:" << param.inputScale
                  << ", input_offset:" << param.inputOffset << ", input_alpha:" << param.inputAlpha;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *QuantOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::QuantParam param;
    param.inputScale = paramJson["input_scale"].get<float>();
    param.inputOffset = paramJson["input_offset"].get<int>();
    ATB_LOG(INFO) << "QuantParam input scale:" << param.inputScale << ", input_offset:" << param.inputOffset;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *SelfAttentionKvCacheFusionOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::SelfAttentionKvCacheFusionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
        ATB_LOG(FATAL) << "token offset:" << param.tokenOffset.at(0);
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
        ATB_LOG(FATAL) << "seqLen:" << param.seqLen.at(0);
    }
    ATB_LOG(INFO) << "SelfAttentionKvCacheFusionParam headNum:" << param.headNum;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *WordEmbeddingParallelOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::WordEmbeddingParallelParam param;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("rankRoot")) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *LmHeadParallelOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::LmHeadParallelParam param;
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("rankRoot")) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *RmsPreNormQuantOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::RmsPreNormQuantParam param;
    param.inputScale = paramJson["inputScale"].get<double>();
    param.inputOffset = paramJson["inputOffset"].get<int>();
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *RmsNormQuantOperationCreate(const nlohmann::json &paramJson)
{
#if 0
    atb::infer_old::RmsNormQuantParam param;
    param.inputScale = paramJson["inputScale"].get<double>();
    param.inputOffset = paramJson["inputOffset"].get<int>();
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
#else
    return nullptr;
#endif
}

static atb::Operation *ActivationOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ActivationParam param;
    if (paramJson.contains("activationType")) {
        param.activationType = 
            atb::infer::ActivationType(paramJson["activationType"].get<int32_t>());
    }
    if (paramJson.contains("scale")) {
        param.scale = paramJson["scale"].get<float>();
    }
    ATB_LOG(INFO) << "ActivationParam activationType:" << param.activationType << ", scale:" << param.scale;
    atb::Operation *op;
    CreateOp(param, &op);
    return op;
}

static atb::Operation *ChatGlm6BLayerEncoderOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::ChatGlm6BLayerEncoderParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    ATB_LOG(INFO) << "ChatGlm6BLayerEncoderParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale;
    atb::Operation *op;
    atb_speed::CreateChatGlm6BLayerEncoderOperation(param, &op);
    return op;
}

static atb::Operation *ChatGlm6BLayeDecoderFlashAttentionOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::ChatGlm6BLayerDecoderFlashAttentionParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    atb::Operation *op;
    atb_speed::CreateChatGlm6BLayerDecoderFlashAttentionOperation(param, &op);
    return op;
}

static atb::Operation *ChatGlm6BLayerDecoderWithoutFusionOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::ChatGlm6BLayerDecoderWithoutFusionParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.transKey = paramJson["transKey"].get<bool>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    param.beginNormAxis = paramJson["beginNormAxis"].get<int>();
    ATB_LOG(INFO) << "ChatGlm6BLayerDecoderParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", transKey:" << param.transKey << ", dk:" << param.dk << ", layerId:" << param.layerId
                  << ", residualAddScale:" << param.residualAddScale << ", beginNormAxis:" << param.beginNormAxis;
    atb::Operation *op;
    atb_speed::ChatGlm6BLayerDecoderWithoutFusionOperation(param, &op);
    return op;
}

static atb::Operation *Chatglm6BLmHeadOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::ChatGlm6BLmHeadParam param;
    ATB_LOG(INFO) << "ChatGlm6BLmHeadParam";
    atb::Operation *op;
    atb_speed::CreateChatGlm6BLmHeadOperation(param, &op);
    return op;
}

static atb::Operation *LlamaLayerEncoderParallelOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::LlamaLayerEncoderParallelParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ATB_LOG(INFO) << "LLaMA65BLayerEncoderParam rmsNormEps:" << param.rmsNormEps << ", headNum:" << param.headNum
                  << ", dk:" << param.dk << ", rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", model:" << param.model;
    atb::Operation *op;
    atb_speed::LlamaLayerEncoderParallelOperation(param, &op);
    return op;
}
 
static atb::Operation *LlamaLayerFusionParallelOperationCreate(const nlohmann::json &paramJson)
{
    atb_speed::LlamaLayerFusionParallelParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    param.layerId = paramJson["layerId"].get<int>();
    param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    for (auto item : paramJson["tokenOffset"]) {
        param.tokenOffset.push_back(item.get<int>());
        ATB_LOG(FATAL) << "token offset:" << param.tokenOffset.at(0);
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
        ATB_LOG(FATAL) << "seqLen:" << param.seqLen.at(0);
    }
    ATB_LOG(INFO) << "LLaMA65BLayerEncoderParam rmsNormEps:" << param.rmsNormEps << ", headNum:" << param.headNum
                  << ", dk:" << param.dk << ", rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", model:" << param.model << ", layerId:" << param.layerId << ", rotaryCoeff:" << param.rotaryCoeff;
    atb::Operation *op;
    atb_speed::LlamaLayerFusionParallelOperation(param, &op);
    return op;
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"RmsPreNormQuantOperation", &RmsPreNormQuantOperationCreate},
    {"RmsNormQuantOperation", &RmsNormQuantOperationCreate},
    {"AllReduceOperation", &AllReduceOperationCreate},
    {"AllGatherOperation", &AllGatherOperationCreate},
    {"LinearParallelOperation", &LinearParallelOperationCreate},
    {"NormOperation", &NormOperationCreate},
    {"RopeOperation", &RopeOperationCreate},
    {"PositionEmbedding1dSplitFusionOperation", &PositionEmbedding1dSplitFusionOperationCreate},
    {"AddNormOperation", &AddNormOperationCreate},
    {"RmsNormOperation", &RmsNormOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"FfnOperation", &FfnOperationCreate},
    {"FfnOldOperation", &FfnOldOperationCreate},
    {"MlpOperation", &MlpOperationCreate},
    {"MlpQuantOperation", &MlpQuantOperationCreate},
    {"EmbeddingOperation", &EmbeddingOperationCreate},
    {"PositionEmbedding1dSplitOperation", &PositionEmbedding1dSplitOperationCreate},
    {"PositionEmbeddingOperation", &PositionEmbeddingOperationCreate},
    {"SelfAttentionKvCacheOperation", &SelfAttentionKvCacheOperationCreate},
    {"SelfAttentionKvCacheFusionOperation", &SelfAttentionKvCacheFusionOperationCreate},
    {"SelfAttentionOperation", &SelfAttentionOperationCreate},
    {"QuantOperation", &QuantOperationCreate},
    {"AddNormQuantOperation", &AddNormQuantOperationCreate},
    {"NormQuantOperation", &NormQuantOperationCreate},
    {"LinearQuantOperation", &LinearQuantOperationCreate},
    {"FfnQuantOperation", &FfnQuantOperationCreate},
    {"LinearActivationOperation", &LinearActivationOperationCreate},
    {"LmHeadParallelOperation", &LmHeadParallelOperationCreate},
    {"WordEmbeddingParallelOperation", &WordEmbeddingParallelOperationCreate},
    {"ActivationOperation", &ActivationOperationCreate},
    {"ChatGlm6BLayerEncoderOperation", &ChatGlm6BLayerEncoderOperationCreate},
    {"ChatGlm6BLayerDecoderFlashAttentionOperation", &ChatGlm6BLayeDecoderFlashAttentionOperationCreate},
    {"ChatGlm6BLayerDecoderWithoutFusionOperation", &ChatGlm6BLayerDecoderWithoutFusionOperationCreate},
    {"LlamaLayerEncoderParallelOperation", &LlamaLayerEncoderParallelOperationCreate},
    {"LlamaLayerFusionParallelOperation", &LlamaLayerFusionParallelOperationCreate},
    {"Chatglm6BLmHeadOperation", &Chatglm6BLmHeadOperationCreate},
};

atb::Operation *CreateOperation(const std::string &opName, const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        ATB_LOG(ERROR) << "not support opName:" << opName;
        return nullptr;
    }

    try {
        return it->second(paramJson);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return nullptr;
}