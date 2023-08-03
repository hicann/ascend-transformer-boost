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
#include "op_desc_json.h"
#include <functional>
#include <map>
#include "asdops/utils/log/log.h"
#include "asdops/params/params.h"
#include "asdops/params/reduce.h"
#include "asdops/params/split.h"

static void AsStridedJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::AsStrided param;
    opDesc.specificParam = param;
}

static void ActivationJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Activation param;
    param.activationType = static_cast<AsdOps::OpParam::Activation::ActivationType>(
        opDescJson["specificParam"]["activationType"].get<int>());
    opDesc.specificParam = param;
}

static void ElewiseJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Elewise param;
    param.elewiseType =
        static_cast<AsdOps::OpParam::Elewise::ElewiseType>(opDescJson["specificParam"]["elewiseType"].get<int>());
    if (opDescJson["specificParam"].contains("varAttr")) {
        param.varAttr = opDescJson["specificParam"]["varAttr"].get<float>();
    }
    if (opDescJson["specificParam"].contains("scale")) {
        param.scale = opDescJson["specificParam"]["scale"].get<float>();
    }
    if (opDescJson["specificParam"].contains("input_scale")) {
        param.input_scale = opDescJson["specificParam"]["input_scale"].get<float>();
    }
    if (opDescJson["specificParam"].contains("input_offset")) {
        param.input_offset = opDescJson["specificParam"]["input_offset"].get<float>();
    }
    opDesc.specificParam = param;
}

static void SplitJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Split param;
    if (opDescJson["specificParam"].contains("splitDim")) {
        param.splitDim = opDescJson["specificParam"]["splitDim"].get<int>();
    }
    if (opDescJson["specificParam"].contains("splitNum")) {
        param.splitNum = opDescJson["specificParam"]["splitNum"].get<int>();
    }
    opDesc.specificParam = param;
}

static void MatMulJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::MatMul param;
    param.transposeA = opDescJson["specificParam"]["transposeA"].get<bool>();
    param.transposeB = opDescJson["specificParam"]["transposeB"].get<bool>();
    opDesc.specificParam = param;
}

static void ReduceJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Reduce param;
    param.reduceType =
        static_cast<AsdOps::OpParam::Reduce::ReduceType>(opDescJson["specificParam"]["reduceType"].get<int>());
    opDesc.specificParam = param;
}

static void ConcatJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Concat param;
    param.concatDim = opDescJson["specificParam"]["concatDim"].get<int>();
    opDesc.specificParam = param;
}

static void ResizeJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Resize param;
    param.alignCorners = opDescJson["specificParam"]["alignCorners"].get<int>();
    param.halfPixelCenters = opDescJson["specificParam"]["halfPixelCenters"].get<int>();
    opDesc.specificParam = param;
}

static void GatherJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Gather param;
    param.batchDims = opDescJson["specificParam"]["batchDims"].get<int>();
    opDesc.specificParam = param;
}

static void BroadcastJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Broadcast param;
    param.broadcastType =
        static_cast<AsdOps::OpParam::Broadcast::BroadcastType>(opDescJson["specificParam"]["broadcastType"].get<int>());
    opDesc.specificParam = param;
}

static void TransdataJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Transdata param;
    param.transdataType =
        static_cast<AsdOps::OpParam::Transdata::TransdataType>(opDescJson["specificParam"]["transdataType"].get<int>());
    {
        const nlohmann::json outCropsValues = opDescJson["specificParam"]["outCrops"];
        const int outCropsSizes = int(outCropsValues.size());
        param.outCrops.resize(outCropsSizes);
        for (int i = 0; i < outCropsSizes; ++i) {
            param.outCrops[i] = outCropsValues[i].get<int>();
        }
    }
    opDesc.specificParam = param;
}

static void NormJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Norm param;
    param.normType = static_cast<AsdOps::OpParam::Norm::NormType>(opDescJson["specificParam"]["normType"].get<int>());
    {
        const nlohmann::json axesValues = opDescJson["specificParam"]["axes"];
        const int axesSizes = int(axesValues.size());
        param.axes.resize(axesSizes);
        for (int i = 0; i < axesSizes; ++i) {
            param.axes[i] = axesValues[i].get<int>();
        }
    }
    param.begin_norm_axis = opDescJson["specificParam"]["begin_norm_axis"].get<int>();
    param.begin_params_axis = opDescJson["specificParam"]["begin_params_axis"].get<int>();
    param.epsilon = opDescJson["specificParam"]["epsilon"].get<float>();
    opDesc.specificParam = param;
}

static void AttentionJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Attention param;
    param.headSize = opDescJson["specificParam"]["headSize"].get<int>();
    {
        const nlohmann::json qSeqLenValues = opDescJson["specificParam"]["qSeqLen"];
        const int qSeqLenSizes = int(qSeqLenValues.size());
        param.qSeqLen.resize(qSeqLenSizes);
        for (int i = 0; i < qSeqLenSizes; ++i) {
            param.qSeqLen[i] = qSeqLenValues[i].get<int>();
        }
    }
    {
        const nlohmann::json kvSeqLenValues = opDescJson["specificParam"]["kvSeqLen"];
        const int kvSeqLenSizes = int(kvSeqLenValues.size());
        param.kvSeqLen.resize(kvSeqLenSizes);
        for (int i = 0; i < kvSeqLenSizes; ++i) {
            param.kvSeqLen[i] = kvSeqLenValues[i].get<int>();
        }
    }
    opDesc.specificParam = param;
}

static void KVCacheJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::KVCache param = {};
    opDesc.specificParam = param;
}

using OpDescSetFunc = std::function<void(const nlohmann::json &, AsdOps::OpDesc &)>;

static const std::map<std::string, OpDescSetFunc> OP_DESC_JSON_FUNC_MAP = {
    {"AsStridedOperation", AsStridedJson}, {"ActivationOperation", ActivationJson}, {"ElewiseOperation", ElewiseJson},
    {"SplitOperation", SplitJson},         {"MatMulOperation", MatMulJson},         {"ReduceOperation", ReduceJson},
    {"ConcatOperation", ConcatJson},       {"ResizeOperation", ResizeJson},         {"GatherOperation", GatherJson},
    {"BroadcastOperation", BroadcastJson}, {"TransdataOperation", TransdataJson},   {"NormOperation", NormJson},
    {"AttentionOperation", AttentionJson}, {"KVCacheOperatoin", KVCacheJson},
};

void JsonToOpDesc(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    std::string opName = opDescJson["opName"].get<std::string>();
    opDesc.opName = opName;
    auto paramFunc = OP_DESC_JSON_FUNC_MAP.find(opName);
    if (paramFunc == OP_DESC_JSON_FUNC_MAP.end()) {
        ASD_LOG(ERROR) << "no opName " << opName;
        return;
    }
    try {
        paramFunc->second(opDescJson, opDesc);
    } catch (const std::exception &e) {
        ASD_LOG(ERROR) << "convert json fail, error:" << e.what();
    }
}