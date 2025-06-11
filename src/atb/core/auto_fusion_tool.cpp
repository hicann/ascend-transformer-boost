/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/core/auto_fusion_tool.h"
#include <fstream>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <cstdio>
#include "atb/operation/operation_base.h"
#include "atb/utils/log.h"
#include "mki/utils/file_system/file_system.h"
namespace atb {
Status CreateAutoFusionTool(atb::GraphParam &graph, atb::AutoFusion **autoFusion)
{
    *autoFusion = new (std::nothrow) AutoFusionTool(graph);
    if (*autoFusion == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

//!
//! \brief auto fusion存储BiShengIR二进制的相关目录
//!
#define AUTOFUSIONSTRPATH(path) std::string(path)

//!
//! \brief BiShengIR工具的安装目录
//!
#define AUTOFUSIONGETBISHENGPATH() std::getenv("BISHENG_INSTALL_PATH")

//!
//! \brief BiShengIR工具的安装目录
//!
typedef void (*TILING_FUNC_GET)(void *);


//!
//! \struct AutoFusionTilingData
//!
//! \brief BishengIR Host 二进制中tiling数据结构
//!
struct AutoFusionTilingData {
    int64_t key{0};
    int64_t mTile{0};
    int64_t nTile{0};
    int64_t kTile{0};
    int64_t processM{0};
    int64_t processN{0};
    int64_t processK{0};
    int64_t splitKSlices{1};

    int64_t swizzleDir{0};
    int64_t swizzleCnt{1};
    int64_t shuffleKType{0};
    int64_t workspaceReuse{0};
    int64_t workspaceBufferNum{2};
    int64_t pTiles{0};
    int64_t ubMaxBitSize{0};
};

//!
//! \struct AutoFusionKernelArgs
//!
//! \brief BishengIR Device二进制中tiling func传入参数的数据结构
//!
struct AutoFusionKernelArgs {
    void *xDevice;
    void *xDeviceDup;
    int64_t offsetX = 0;
    int64_t sizeX0 = -1;
    int64_t sizeX1 = -1;
    int64_t strideX0 = -1;
    int64_t strideX1 = -1;

    void *yDevice;
    void *yDeviceDup;
    int64_t offsetY = 0;
    int64_t sizeY0 = -1;
    int64_t sizeY1 = -1;
    int64_t strideY0 = -1;
    int64_t strideY1 = -1;

    void *vDevice;
    void *vDeviceDup;
    int64_t offsetV = 0;
    int64_t sizeV0 = -1;
    int64_t sizeV1 = -1;
    int64_t strideV0 = -1;
    int64_t strideV1 = -1;

    void *oDevice;
    void *oDeviceDup;
    int64_t offsetO = -1;
    int64_t sizeO0 = -1;
    int64_t sizeO1 = -1;
    int64_t strideO0 = -1;
    int64_t strideO1 = -1;

    void *tilingDevice;
    void *tilingDeviceDup;
    int64_t offsetTiling = 0;
    int64_t sizeTiling = sizeof(AutoFusionTilingData);
    int64_t strideTiling = 1;
};

AutoFusionTool::AutoFusionTool(atb::GraphParam &graph) : graph_(graph)
{
    homePath_ = std::string(std::getenv("HOME"));
    homePath_ += AUTOFUSIONSTRPATH("/.atb_auto_fusion/bishengir_bin/");
}

void AutoFusionTool::SetFusionClass(const std::set<std::string> &fusionClassArray)
{
    fusionClassMap_ = fusionClassArray;
    return;
}

void AutoFusionTool::genAllTensorIDs(std::set<uint32_t> &allTensorIds, std::set<uint32_t> &allOutTensorIds)
{
    size_t nodeSize = graph_.nodes.size();
    for (size_t i = 0; i < nodeSize; i++) {
        const auto &node = graph_.nodes.at(i);
        const auto &inTensorIds = node.inTensorIds;
        const auto &outTensorIds = node.outTensorIds;
        for (auto id : inTensorIds) {
            allTensorIds.insert(id);
        }
        for (auto id : outTensorIds) {
            allTensorIds.insert(id);
            allOutTensorIds.insert(id);
        }
    }
}

void AutoFusionTool::DoAutoFusion(const std::set<std::string> &fusionClassArray)
{
    std::set<uint32_t> allTensorIds;
    std::set<uint32_t> outTensorIds;
    genAllTensorIDs(allTensorIds, outTensorIds);
    if (fusionClassArray.empty()) {
        return ;
    }
    SetFusionClass(fusionClassArray);
    GetFusionBinAndUpdateFusedGraph();
    std::vector<Node> nodesNew;
    size_t nodeSize = graph_.nodes.size();
    for (size_t i =  0; i < nodeSize; i++) {
        const auto &node = graph_.nodes.at(i);
        const auto &inTensorIds = node.inTensorIds;
        const auto &outTensorIds = node.outTensorIds;
        if (!(inTensorIds.size() == 0 && outTensorIds.size() == 0)) {
            nodesNew.push_back(node);
        }
    }
    graph_.nodes = nodesNew;
    std::set<uint32_t> allTensorIdsNew;
    std::set<uint32_t> allOutTensorIdsNew;
    genAllTensorIDs(allTensorIdsNew, allOutTensorIdsNew);
    updateAllTensorIDs(outTensorIds, allOutTensorIdsNew);
}

void AutoFusionTool::updateAllTensorIDs(const std::set<uint32_t> &outTensorIds, const std::set<uint32_t> &allOutTensorIdsNew)
{
    auto it = outTensorIds.begin();
    auto itNew = allOutTensorIdsNew.begin();
    while (itNew != outTensorIds.end() && *it == *itNew) {
        itNew++;
        it++;
    }
    std::unordered_map<uint32_t, uint32_t> chageMap;
    int interErsed = outTensorIds.size() - allOutTensorIdsNew.size();
    graph_.internalTensorNum -= interErsed;
    while (itNew != allOutTensorIdsNew.end()) {
        chageMap[*itNew] = *it;
        itNew++;
        it++;
    }
    size_t nodeSize = graph_.nodes.size();
    for (size_t i = 0; i < nodeSize; i++) {
        auto &node = graph_.nodes.at(i);
        auto &inTensorIds = node.inTensorIds;
        auto &outTensorIds = node.outTensorIds;
        for (auto &id : inTensorIds) {
            if (chageMap.find(id) != chageMap.end()) {
                id = chageMap[id];
            }
        }
        for (auto &id : outTensorIds) {
            if (chageMap.find(id) != chageMap.end()) {
                id = chageMap[id];
            }
        }
    }
}

void AutoFusionTool::findNodesWhichInputsIs(const uint32_t id, std::vector<uint32_t> &re)
{
    size_t counter = graph_.nodes.size();
    for (size_t i = 0; i < counter; i++) {
        auto inTensorIds = graph_.nodes[i].inTensorIds;
        for (auto inId : inTensorIds) {
            if (inId == id) {
                re.push_back(i);
                break;
            }
        }
    }
    return ;
}

void AutoFusionTool::parseCollectNodes(std::vector<std::pair<std::string, std::vector<uint32_t>>> &fusionclassAndIndex)
{
    for (auto i  : linearNodes_) {
        const auto &outTensorIds = graph_.nodes[i].outTensorIds;
        std::vector<uint32_t> re;
        findNodesWhichInputsIs(outTensorIds[0], re);
        if (re.size() != 1) {
            linearNodes_.erase(linearNodes_.find(i));
            continue;
        } else {
            // here
            uint32_t key = re[0];
            bool flagAdd = (eleAddNodes_.find(key) != eleAddNodes_.end());
            bool flagGelu = (actGeluNodes_.find(key) != actGeluNodes_.end());
            bool flagSigmoid = (actSigmoidNodes_.find(key) != actSigmoidNodes_.end());
            bool flagSwiglu = (actSwiGluNodes_.find(key) != actSwiGluNodes_.end());
            if (true == flagAdd) {
                fusionclassAndIndex.push_back(
                    std::make_pair<std::string, std::vector<uint32_t>>("matmul_add", {i, key}));
            } else if (true == flagGelu) {
                fusionclassAndIndex.push_back(
                    std::make_pair<std::string, std::vector<uint32_t>>("matmul_gelu", {i, key}));
            } else if (true == flagSigmoid) {
                fusionclassAndIndex.push_back(
                    std::make_pair<std::string, std::vector<uint32_t>>("matmul_sigmoid", {i, key}));
            } else if (true == flagSwiglu) {
                fusionclassAndIndex.push_back(
                    std::make_pair<std::string, std::vector<uint32_t>>("matmul_swiglu", {i, key}));
            }
        }
    }
    for (auto i : fusionclassAndIndex) {
        ATB_LOG(INFO) << "fusion type = " << i.first;
        ATB_LOG(INFO) << "{";
        for (auto j : i.second) {
            ATB_LOG(INFO) << j << ",";
        }
        ATB_LOG(INFO) << "}";
    }
    return ;
}

void AutoFusionTool::CollectNodes()
{
    const auto &nodes = graph_.nodes;
    bool ismatmulAdd = (fusionClassMap_.find("matmul_add") != fusionClassMap_.end());
    bool ismatmulGelu = (fusionClassMap_.find("matmul_gelu") != fusionClassMap_.end());
    if (!ismatmulAdd && !ismatmulGelu) {
        return;
    }
    // here
    size_t counter = nodes.size();
    for (size_t i = 0; i < counter; i++) {
        OperationBase *opBase = dynamic_cast<OperationBase *>(nodes[i].operation);
        nlohmann::json paramJson = opBase->GetParamJson();
        size_t inTensorSize = nodes[i].inTensorIds.size();
        size_t outTensorSize = nodes[i].outTensorIds.size();
        if (opBase->GetName() == "LinearOperation" && inTensorSize == 2 &&
            outTensorSize == 1) {
            if (paramJson["matmulType"] == 0) {
                linearNodes_.insert(i);
            }
        }
        if (ismatmulAdd == true && opBase->GetName() == "ElewiseOperation" && inTensorSize == 2 &&
            outTensorSize == 1) {
            if (paramJson["elewiseType"] == atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD) {
                eleAddNodes_.insert(i);
            }
        }
        if (ismatmulGelu == true && opBase->GetName() == "ActivationOperation" && inTensorSize == 1 &&
            outTensorSize == 1) {
            if (paramJson["activationType"] == atb::infer::ActivationType::ACTIVATION_GELU) {
                actGeluNodes_.insert(i);
            }
        }
    }
    return;
}

void AutoFusionTool::ParseFusion(std::vector<std::pair<std::string, std::vector<uint32_t>>> &fusionclassAndIndex)
{
    CollectNodes();
    return parseCollectNodes(fusionclassAndIndex);
}

void AutoFusionTool::GetFusionBinAndUpdateFusedGraph()
{
    std::vector<std::pair<std::string, std::vector<uint32_t>>> fusionclassAndIndex;
    ParseFusion(fusionclassAndIndex);
    std::vector<std::pair<std::string, std::string>> fusionclassAndBin;
    // 保留
    for (auto i : fusionclassAndIndex) {
        auto path = callBiShengIR(i) ? homePath_ : "";
        fusionclassAndBin.push_back(std::make_pair(i.first, path));
    }

    UpdateFusedGraph(fusionclassAndIndex, fusionclassAndBin);
    return;
}

bool AutoFusionTool::callBiShengIR(const std::pair<std::string, std::vector<uint32_t>> &subFusion)
{
    // 以mlir为入手点
    std::string fusionClass = subFusion.first;
    std::string hostBinPath = homePath_ + "lib" + fusionClass + ".so";
    // here
    std::string preFill = homePath_ + fusionClass;
    std::string deviceBinPath = preFill + ".o";
    std::string jsonPath = preFill + ".json";
    std::string cppPath = preFill + ".cpp";
    if (Mki::FileSystem::Exists(hostBinPath) && Mki::FileSystem::Exists(deviceBinPath) &&
        Mki::FileSystem::Exists(jsonPath) && Mki::FileSystem::Exists(cppPath)) {
        return true;
    }
    if (!AUTOFUSIONGETBISHENGPATH()) {
        return false;
    }
    if (!Mki::FileSystem::Exists(homePath_)) {
        std::string cmd = "mkdir -p " + homePath_;
        auto ret = system(cmd.c_str());
        if (ret != 0) {
            ATB_LOG(ERROR) << "BishengIR folder create failed!";
            return false;
        }
    }
    genMlirAndBin(fusionClass);
    genFusionKernelDef(fusionClass);
    return true;
}

void AutoFusionTool::UpdateFusedGraph(std::vector<std::pair<std::string, std::vector<uint32_t>>> &fusionclassAndIndex,
    const std::vector<std::pair<std::string, std::string>> &fusionclassAndBin)
{
    size_t counter = fusionclassAndBin.size();
    for (size_t i = 0; i < counter; i++) {
        if (fusionclassAndBin[i].second == "") {
            continue;
        }
        SortNodeByTensorIdsAndChangeTopology(fusionclassAndIndex[i].second, fusionclassAndIndex[i].first);
    }
    return ;
}

void AutoFusionTool::SortNodeByTensorIdsAndChangeTopology(std::vector<uint32_t> &fusedNodes, const std::string &fusionClass)
{
    std::sort(fusedNodes.begin(), fusedNodes.end(), [this](int index1, int index2) {
        bool flag = false;
        for (auto &subIndex : graph_.nodes[index2].inTensorIds) {
            for (auto &subIndex1 : graph_.nodes[index1].outTensorIds) {
                if (subIndex1 == subIndex && uint32_t(-1) != subIndex1) {
                    subIndex1 = -1;
                    subIndex = -1;
                    flag = true;
                }
            }
        }
        if (flag) {
            return flag;
        }
        for (auto &subIndex : graph_.nodes[index1].inTensorIds) {
            for (auto &subIndex1 : graph_.nodes[index2].outTensorIds) {
                if (subIndex1 == subIndex && uint32_t(-1) != subIndex1) {
                    subIndex1 = -1;
                    subIndex = -1;
                    flag = false;
                }
            }
        }
        return flag;
    });
    UpdateGraphStruct(fusedNodes, fusionClass);
    return;
}

void AutoFusionTool::SetFusionParam(atb::infer::FusionParam &param, const std::string &fusionClass)
{
    if (fusionClass == "matmul_add") {
        param.fusionType = atb::infer::FusionParam::MATMUL_ADD;
    } else if (fusionClass == "matmul_gelu") {
        param.fusionType = atb::infer::FusionParam::MATMUL_GELU;
    } else if (fusionClass == "matmul_sigmoid") {
        param.fusionType = atb::infer::FusionParam::MATMUL_SIGMOID;
    } else if (fusionClass == "matmul_swiglu") {
        param.fusionType = atb::infer::FusionParam::MATMUL_SWIGLU;
    } else {
        param.fusionType = atb::infer::FusionParam::NON_FUSION;
    }
    return;
}

void AutoFusionTool::UpdateGraphStruct(const std::vector<uint32_t> &fusedNodes, const std::string &fusionClass)
{
    auto &preInTensorIds = graph_.nodes[fusedNodes[0]].inTensorIds;
    auto &preOutTensorIds = graph_.nodes[fusedNodes[0]].outTensorIds;
    size_t counter = fusedNodes.size();
    for (size_t i = 1; i < counter; i++) {
        auto &currentInTensorIds = graph_.nodes[fusedNodes[i]].inTensorIds;
        for (auto subIndex : currentInTensorIds) {
            if (subIndex != uint32_t(-1)) {
                preInTensorIds.push_back(subIndex);
            }
        }
        preOutTensorIds = (graph_.nodes[fusedNodes[i]].outTensorIds);
    }
    atb::infer::FusionParam param;
    SetFusionParam(param, fusionClass);
    // here
    UpdateReshapeFunc(fusedNodes);
    atb::Operation *rawPtr = nullptr;
    CreateOperation(param, &rawPtr);
    if (rawPtr == nullptr) {
        throw std::runtime_error("add operation is null.");
    }
    if (graph_.nodes[fusedNodes[0]].operation) {
        delete graph_.nodes[fusedNodes[0]].operation;
        graph_.nodes[fusedNodes[0]].operation = nullptr;
    }
    graph_.nodes[fusedNodes[0]].operation = rawPtr;
    counter = fusedNodes.size();
    for (size_t i = 1; i < counter; i++) {
        ATB_LOG(INFO) << "null fusion operation created";
        atb::infer::FusionParam param;
        atb::Operation *rawPtr = nullptr;
        CreateOperation(param, &rawPtr);
        if (rawPtr == nullptr) {
            throw std::runtime_error("add operation is null.");
        }
        if (graph_.nodes[fusedNodes[i]].operation) {
            delete graph_.nodes[fusedNodes[i]].operation;
            graph_.nodes[fusedNodes[i]].operation = nullptr;
        }
        graph_.nodes[fusedNodes[i]].operation = rawPtr;
        graph_.nodes[fusedNodes[i]].inTensorIds.clear();
        graph_.nodes[fusedNodes[i]].outTensorIds.clear();
    }
    ATB_LOG(INFO) << " UpdateGraphStruct Done!";
}

void AutoFusionTool::UpdateReshapeFunc(const std::vector<uint32_t> &fusedNodes)
{
    auto &inTensorReshapeFuncs = graph_.nodes[fusedNodes[0]].inTensorReshapeFuncs;
    size_t counter = fusedNodes.size();
    for (size_t i = 1; i < counter; i++) {
        size_t funcsCounter = graph_.nodes[fusedNodes[i]].inTensorReshapeFuncs.size();
        for (size_t j = 1; j < funcsCounter; j++) {
            inTensorReshapeFuncs.push_back(graph_.nodes[fusedNodes[i]].inTensorReshapeFuncs[j]);
        }
    }
    return ;
}

std::string AutoFusionTool::getTilingKey(const std::string &fusionClass)
{
    std::string path = homePath_;
    if ("matmul_add" == fusionClass) {
        path += "libmatmul_add.so";
    } else if ("matmul_gelu" == fusionClass) {
        path += "libmatmul_gelu.so";
    } else if ("matmul_sigmoid" == fusionClass) {
        path += "libmatmul_sigmoid.so";
    } else if ("matmul_swiglu" == fusionClass) {
        path += "libmatmul_swiglu.so";
    }
    AutoFusionTilingData *tilingDataPtr = new AutoFusionTilingData;
    void *handle = dlopen(path.c_str(), RTLD_LAZY);
    if (!handle) {
        ATB_LOG(ERROR) << "host tiling load error!";
    }
    TILING_FUNC_GET tiling_func = nullptr;
    std::string tiling_func_name = fusionClass + "_tiling_func";
    *(void **)(&tiling_func) = dlsym(handle, tiling_func_name.c_str());
    AutoFusionKernelArgs *autoFusionKernelArgs = new AutoFusionKernelArgs;
    autoFusionKernelArgs->tilingDevice = static_cast<void *>(tilingDataPtr);
    autoFusionKernelArgs->tilingDeviceDup = autoFusionKernelArgs->tilingDevice;

    tiling_func((void *)autoFusionKernelArgs);
    std::string key = std::to_string(tilingDataPtr->key);
    delete tilingDataPtr;
    delete autoFusionKernelArgs;
    return key;
}
bool AutoFusionTool::genFusionKernelDef(const std::string &fusionClass)
{
    nlohmann::json fusionJson{{"binFileName", "matmul_add"},
        {"binFileSuffix", ".o"},
        {"blockDim", 40},
        {"coreType", "MIX"},
        {"core_type", "MIX"},
        {"intercoreSync", 0},
        {"magic", "RT_DEV_BINARY_MAGIC_ELF"},
        {"memoryStamping", {}},
        {"opParaSize", 0},
        {"parameters", {}},
        {"sha256", ""},
        {"kernelList", {}},
        {"compileInfo", nlohmann::json::object()}};
    fusionJson["binFileName"] = nlohmann::json::array({});
    fusionJson["parameters"] = nlohmann::json::array({});
    fusionJson["memoryStamping"] = nlohmann::json::array({});
    fusionJson["kernelList"] = nlohmann::json::array({});
    std::string key = getTilingKey(fusionClass);
    if (key != "") {
        key = "_" + key;
    }
    if ("matmul_add" == fusionClass) {
        fusionJson["binFileName"] = "matmul_add";
        nlohmann::json JObject = {{"kernelName", "matmul_add" + key}};
        fusionJson["kernelList"].push_back(JObject);
    } else if ("matmul_gelu" == fusionClass) {
        fusionJson["binFileName"] = "matmul_gelu";
        nlohmann::json JObject = {{"kernelName", "matmul_gelu" + key}};
        fusionJson["kernelList"].push_back(JObject);
    } else if ("matmul_sigmoid" == fusionClass) {
        fusionJson["binFileName"] = "matmul_sigmoid";
        nlohmann::json JObject = {{"kernelName", "matmul_sigmoid" + key}};
        fusionJson["kernelList"].push_back(JObject);
    } else if ("matmul_swiglu" == fusionClass) {
        fusionJson["binFileName"] = "matmul_swiglu";
        nlohmann::json JObject = {{"kernelName", "matmul_swiglu" + key}};
        fusionJson["kernelList"].push_back(JObject);
    }
    std::string jsonPath = homePath_ + fusionClass + ".json";
    std::ofstream file(jsonPath);
    file << fusionJson.dump(4);
    file.close();
    std::string pyImport = "import os\nimport configparser\nimport json\nimport struct\nimport logging\n";
    std::string aligned = "def aligned_string(s:str, align:int) -> str:\n\
    width = (len(s) // align + 1) * align\n\
    return s.ljust(width, '\\0')\n";
    std::string pyGetHeader =
        "def get_header_from_file(file_path):\n    result = True\n    magic_dict = {\"RT_DEV_BINARY_MAGIC_ELF\": 0x43554245,\"RT_DEV_BINARY_MAGIC_ELF_AIVEC\": 0x41415246,\"RT_DEV_BINARY_MAGIC_ELF_AICUBE\": 0x41494343}\n    core_type_dict = {\"AiCore\": 0, \"VectorCore\": 2, \"MIX\": 4}\n    aling_bytes = struct.calcsize('I')\n    fixed_header_len = 128\n    header = b''\n    try:\n        with open(file_path) as f:\n            text = json.load(f) \n\
            version = 0  \n\
            crc = 0  \n\
            compile_info_str = aligned_string(json.dumps(text[\"compileInfo\"]), aling_bytes) \n\
            op_para_size = text[\"opParaSize\"]  \n\
            core_type = core_type_dict.get(text[\"coreType\"], 0)  \n\
            magic_type = text[\"magic\"]  \n\
            if magic_type not in magic_dict: \n\
                logging.error(\"magic %s is invalid\", magic_type)  \n\
                result = False  \n\
            else: \n\
                magic = magic_dict[magic_type] \n\
            kernel_list = [] \n\
            if \"kernelList\" in text: \n\
                for kernel_item in text[\"kernelList\"]: \n\
                    kernel_list.append(aligned_string(kernel_item[\"kernelName\"], aling_bytes)) \n\
            else: \n\
                kernel_list.append(aligned_string(text[\"kernelName\"], aling_bytes)) \n\
            kernel_num = len(kernel_list) \n\
            if kernel_num == 0: \n\
                result = False \n\
            header = struct.pack('I', version) + struct.pack('I', magic) + struct.pack('I', op_para_size) + struct.pack('I', core_type) + struct.pack('I', kernel_num) \n\
            offset = 0 \n\
            kernel_name_offset = offset \n\
            for kernel_name in kernel_list: \n\
                offset += (aling_bytes + len(kernel_name)) \n\
            compile_info_offset = offset \n\
            offset += (aling_bytes + len(compile_info_str)) \n\
            binary_offset = offset \n\
            header = header + struct.pack('I', kernel_name_offset) + struct.pack('I', compile_info_offset) + struct.pack('I', binary_offset) \n\
            intercore_sync = text.get(\"intercoreSync\", 0) \n\
            task_ration_type = text.get(\"taskRation\", \"tilingKey\") \n\
            if task_ration_type == \"tilingKey\": \n\
                task_ration = 0 \n\
            else: \n\
                ration = [int(r) for r in task_ration_type.split(\":\")] \n\
                if len(ration) != 2: \n\
                    logging.error(f\"ration is invalid: {task_ration_type}\") \n\
                    result = False \n\
                task_ration = (ration[0] << 16) + ration[1] \n\
            header = header + struct.pack('I', intercore_sync) + struct.pack('I', task_ration) \n\
            header = header.ljust(fixed_header_len - aling_bytes, b'\\x00') \n\
            header += struct.pack('I', crc) \n\
            for kernel_name in kernel_list: \n\
                header += struct.pack('I', len(kernel_name)) \n\
                header += kernel_name.encode('utf-8') \n\
            header += struct.pack('I', len(compile_info_str)) \n\
            header += compile_info_str.encode('utf-8') \n\
    except FileNotFoundError: \n\
        logging.error(\"file %s is not found!\", file_path) \n\
        result = False \n\
    except json.decoder.JSONDecodeError: \n\
        logging.error(\"file %s is not json file!\", file_path) \n\
        result = False \n\
    except KeyError: \n\
        logging.error(\"keyerror in file %s!\", file_path) \n\
        result = False \n\
    return header, result\n";
    // here
    std::string pyWriteCpp =
        "def write_to_cpp(binary_path, header, dst_cpp_path, kernel, target_version, is_const=True):\n    try:\n       "
        " with open(binary_path, 'rb') as f:\n            data = f.read()\n            binary_size = len(data)\n       "
        "     header += struct.pack('I', binary_size)\n            data = header + data\n    except "
        "FileNotFoundError:\n        logging.error(\"file %s is not found!\", binary_path)\n        return False\n    "
        "name = f'KERNELBIN_{kernel.upper()}_{target_version.upper()}'\n    data_type = 'const uint8_t' if is_const "
        "else 'uint8_t'\n    with open(dst_cpp_path, 'w') as f:\n        for i in range(0, len(data), 1):\n            "
        "f.write(''.join('{:02x}'.format(b) for b in data[i:i+1]))\n            f.write('\\n')\n    return True\n";

    std::string finalPy = "def compile_ascendc_code(obj_path, dst_cpp_path, is_const=True):\n\
    if not obj_path.endswith('.o'): \n\
        logging.error(\"%s is not an obj file.\", obj_path) \n\
        exit(1) \n\
    json_file = obj_path.rsplit('.', 1)[0] + '.json' \n\
    header, result = get_header_from_file(json_file) \n\
    if not result: \n\
        logging.error(\"failed to parse file %s.\", json_file) \n\
        exit(1) \n\
    obj_realpath = os.path.realpath(obj_path) \n\
    kernel = obj_path.split('/')[-2] \n\
    target_version = obj_realpath.split('/')[-4] \n\
    output_dir = os.path.dirname(dst_cpp_path) \n\
    if not os.path.exists(output_dir): \n\
        os.makedirs(output_dir, exist_ok=True) \n\
    result = write_to_cpp(obj_path, header, dst_cpp_path, kernel, target_version, is_const) \n\
    if not result: \n\
        logging.error(\"failed to write into file %s.\", dst_cpp_path) \n\
        exit(1) \n";
    std::string cmd1 = "compile_ascendc_code(\"" + homePath_ + fusionClass +
                       ".o\","
                       "\"" +
                       homePath_ + fusionClass + ".cpp" + "\")";
    std::string cmd2 = "python3 " + homePath_ + "genCpp.py";
    std::string pypath = homePath_ + "genCpp.py";
    std::ofstream outfile;
    outfile.open(pypath.c_str(), std::ios::out | std::ios::trunc);
    outfile << pyImport << aligned << pyGetHeader << pyWriteCpp << finalPy << cmd1;
    outfile.close();
    auto ret = system(cmd2.c_str());
    if (ret != 0) {
        return false;
    }
    return true;
}


bool AutoFusionTool::genMlirAndBin(const std::string &fusionClass)
{
    int res = -1;
    if ("matmul_add" == fusionClass) {
        std::string matmul_add_mlir = "module { \
            func.func @matmul_add(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>, %arg3: tensor<?x?xf16>) -> tensor<?x?xf16> \
            attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>} { \
            %c0 = arith.constant 0 : index \
            %c1 = arith.constant 0 : index \
            %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16> \
            %dim2 = tensor.dim %arg1, %c1 : tensor<?x?xf16> \
            %0 = tensor.empty(%dim, %dim2) : tensor<?x?xf16> \
            %1 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg3 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            return %2 : tensor<?x?xf16> \
            } \
        }";
        std::ofstream outfile;
        auto mlirPath = homePath_ + "matmul_add.mlir";
        outfile.open(mlirPath.c_str(), std::ios::out | std::ios::trunc);
        outfile << matmul_add_mlir;
        outfile.close();
        std::string matmul_add_bash =
            "#!/bin/bash\npath=$HOME'/.atb_auto_fusion/bishengir_bin/'\narch=$(uname -i)\nif [ \"$arch\" != "
            "\"aarch64\" ]; then\n     arch=\'x86\'\nfi\necho "
            "\'Arch is \'$arch\', BishengIR \'$arch\' should installed ...\'\nBISHENG_PATH=$BISHENG_PATH\necho "
            "$BISHENG_PATH\nbishengir_opt=$BISHENG_PATH$\'bishengir_\'$arch\'/bin/"
            "bishengir-opt\'\nbishengir_compile=$BISHENG_PATH$\'bishengir_\'$arch\'/bin/bishengir-compile\'\nexport "
            "LD_LIBRARY_PATH=$BISHENG_PATH:$LD_LIBRARY_PATH\nexport "
            "PATH=$BISHENG_PATH/ccec_compiler_$arch/bin:$PATH\nexport "
            "BISHENG_INSTALL_PATH=$BISHENG_PATH/ccec_compiler_$arch/bin\necho $bishengir_opt\necho "
            "$bishengir_compile\n$bishengir_opt -lower-hfusion-pipeline=\"block-dim=40 enable-workspace-reuse=true\" "
            "-convert-hfusion-to-hivm -cse $path/matmul_add.mlir -o $path/matmul_add_lower.mlir\n$bishengir_compile "
            "-enable-lir-compile=true -enable-hfusion-compile=false -enable-hivm-compile=true "
            "-enable-multi-kernel-compile $path/matmul_add_lower.mlir -o $path/matmul_add";
        mlirPath = homePath_ + "gen_matmul_add_bin.sh";
        outfile.open(mlirPath.c_str(), std::ios::out | std::ios::trunc);
        outfile << matmul_add_bash;
        outfile.close();
        res = system(("bash " + homePath_ + AUTOFUSIONSTRPATH("gen_matmul_add_bin.sh")).c_str());
    } else if ("matmul_gelu" == fusionClass) {
        std::string matmul_gelu_mlir = "module { \
            func.func @matmul_gelu(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16>) -> tensor<?x?xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>} {\
            %c0 = arith.constant 0 : index \
            %c1 = arith.constant 0 : index \
            %c2 = arith.constant 1 : index \
            %cst = arith.constant 4.470830e-02 : f16 \
            %cst_0 = arith.constant -1.595700e+00 : f16 \
            %cst_1 = arith.constant 1.000000e+00 : f16 \
            %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16> \
            %dim2 = tensor.dim %arg1, %c1 : tensor<?x?xf16> \
            %0 = tensor.empty(%dim, %dim2) : tensor<?x?xf16> \
            %1 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %dim_2 = tensor.dim %1, %c0 : tensor<?x?xf16> \
            %dim_3 = tensor.dim %1, %c2 : tensor<?x?xf16>  \
            %2 = tensor.empty(%dim_2, %dim_3) : tensor<?x?xf16> \
            %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %cst : tensor<?x?xf16>, f16) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %cst_0 : tensor<?x?xf16>, f16) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%7 : tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %cst_1 : tensor<?x?xf16>, f16) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%1, %9 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16> \
            return %10 : tensor<?x?xf16> \
            } \
            }";
        std::ofstream outfile;
        auto mlirPath = homePath_ + "matmul_gelu.mlir";
        outfile.open(mlirPath.c_str(), std::ios::out | std::ios::trunc);
        outfile << matmul_gelu_mlir;
        outfile.close();
        std::string matmul_gelu_bash =
            "#!/bin/bash\npath=$HOME'/.atb_auto_fusion/bishengir_bin/'\narch=$(uname -i)\nif [ \"$arch\" != "
            "\"aarch64\" ]; then\n arch=\'x86\'\nfi\necho "
            "\'Arch is \'$arch\', BishengIR \'$arch\' should installed ...\'\nBISHENG_PATH=$BISHENG_PATH\necho "
            "$BISHENG_PATH\nbishengir_opt=$BISHENG_PATH$\'bishengir_\'$arch\'/bin/"
            "bishengir-opt\'\nbishengir_compile=$BISHENG_PATH$\'bishengir_\'$arch\'/bin/bishengir-compile\'\nexport "
            "LD_LIBRARY_PATH=$BISHENG_PATH:$LD_LIBRARY_PATH\nexport "
            "PATH=$BISHENG_PATH/ccec_compiler_$arch/bin:$PATH\nexport "
            "BISHENG_INSTALL_PATH=$BISHENG_PATH/ccec_compiler_$arch/bin\necho $bishengir_opt\necho "
            "$bishengir_compile\n$bishengir_opt -lower-hfusion-pipeline=\"block-dim=40 enable-workspace-reuse=true\" "
            "-convert-hfusion-to-hivm -cse $path/matmul_gelu.mlir -o $path/matmul_gelu_lower.mlir\n$bishengir_compile "
            "-enable-lir-compile=true -enable-hfusion-compile=false -enable-hivm-compile=true "
            "-enable-multi-kernel-compile $path/matmul_gelu_lower.mlir -o $path/matmul_gelu";
        mlirPath = homePath_ + "gen_matmul_gelu_bin.sh";
        outfile.open(mlirPath.c_str(), std::ios::out | std::ios::trunc);
        outfile << matmul_gelu_bash;
        outfile.close();
        res = system(("bash " + homePath_ + AUTOFUSIONSTRPATH("gen_matmul_gelu_bin.sh")).c_str());
    }
    if (res != 0) {
        return false;
    }
    return true;
}
} // namespace atb