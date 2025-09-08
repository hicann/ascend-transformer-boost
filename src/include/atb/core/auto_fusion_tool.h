/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTO_FUSION_TOOL_H
#define AUTO_FUSION_TOOL_H
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <set>
#include "atb/auto_fusion.h"
#include "atb/context.h"
#include "atb/graph_op_builder.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#include "atb/operation.h"
#include "atb/svector.h"
#include "atb/types.h"
#include "atb/utils.h"

//!
//! \file auto_fusion_tool.h
//!
//! \brief 定义算子自动融合AutoFusion类
//!
namespace atb {
//!
//! \class AutoFusionTool
//!
//! \brief 定义算子自动融合AutoFusion类
//!
//! 该接口类定义了算子自动融合工具类，提供了自动融合的函数方法。
//!
    
class AutoFusionTool : public AutoFusion {
public:
    //! \brief 构造函数
    //!
    //! \param graph AutoFusion类被创建的时候需要绑定的图结构
    //!
    AutoFusionTool(atb::GraphParam &graph);

    //! \brief 析构函数。
    ~AutoFusionTool() = default;

    //! \brief 执行auto fusion的入口函数
    //!
    //! \param fusionClassArray 用户指定的融合类型，默认是空，表示不触发算子自动融合
    //!
    void DoAutoFusion(const std::set<std::string> &fusionClassArray = {}) override;
private:
    //! \brief 判断id为传入参数的tensor是不是其他node的输入
    //!
    //! \param id 被判断的tensor id
    //! \param re 返回的结果
    void findNodesWhichInputsIs(const uint32_t id, std::vector<uint32_t> &re);

    //! \brief 针对手机到的node进行分析，筛选出可以被融合的nodes
    //! \param fusionclassAndIndex 存放结果的容器
    void parseCollectNodes(std::vector<std::pair<std::string, std::vector<uint32_t>>> &fusionclassAndIndex);

    //! \brief 收集符合融合条件的nodes
    //!
    void CollectNodes();

    //! \brief 使用dialect对图进行描述，根据描述所得的各个node的名称+输入输出的
    //! 拓扑逻辑给出能否进行融合的判断，输出std::pair<std::string, vector<int>>数组
    //! \param fusionclassAndIndex 需要更新的图结构中的node pair所对应的融合之后算子的存储路径
    //!
    void ParseFusion(std::vector<std::pair<std::string, std::vector<uint32_t>>> &fusionclassAndIndex);


    //! \brief 根据上述得到的vector<pair<string, vector<int>>>, 对每个pair获取BiShengIR的所编译出来的算子二进制
    //!
    void GetFusionBinAndUpdateFusedGraph();

    //! \brief 更新自动融合之后的算子，并且针对node struct增加融合标签fused，在后续的推理获取kernel的过程中，实现动静分流
    //!
    //! \param fusionclassAndIndex 需要更新的图结构中的融合类型+node pair
    //! \param fusionclassAndBin 需要更新的图结构中的融合类型+二进制路径pair
    //!
    void UpdateFusedGraph(std::vector<std::pair<std::string, std::vector<uint32_t>>> &fusionclassAndIndex,
        const std::vector<std::pair<std::string, std::string>> &fusionclassAndBin);

    //! \brief 针对输入输出的tensor的ids对对nodes进行排序，并且直接输出连接node的tensor的id vector，合并2个node的输入输出
    //!
    //! \param fusedNodes 自动融合node数组
    //! \param fusionClass 自动融合类型
    //!
    void SortNodeByTensorIdsAndChangeTopology(std::vector<uint32_t> &fusedNodes,
        const std::string &fusionClass);

    //! \brief 根据排序结构更新图结构
    //!
    //! \param fusedNodes 自动融合node数组
    //! \param fusionClass 自动融合类型
    //!
    void UpdateGraphStruct(const std::vector<uint32_t> &fusedNodes,
        const std::string &fusionClass);

    //! \brief 更新融合之后的reshape func
    //!
    //! \param fusedNodes 自动融合node数组
    //!
    void UpdateReshapeFunc(const std::vector<uint32_t> &fusedNodes);

    //! \brief 调用BiShengIR， 生成二进制
    //!
    //! \param subFusion 自动融合类型-nodes pair
    //! \return 是否正常生成的二进制的存储路径
    //!
    bool callBiShengIR(const std::pair<std::string, std::vector<uint32_t>> &subFusion);

    //! \brief 生成二进制
    //!
    //! \param fusionClass 融合类型
    //! \return 生成是否成功
    //!
    bool genMlirAndBin(const std::string &fusionClass);

    //! \brief 生成融合算子定义
    //!
    //! \param fusionClass 融合类型
    //! \return 生成是否成功
    //!
    bool genFusionKernelDef(const std::string &fusionClass);

    //! \brief 获得当前图中所有的tensor id以及所有的输出tensor id
    //!
    //! \param allTensorIds 所有张量
    //! \param allOutTensorIds 所有输出张量
    //!
    void genAllTensorIDs(std::set<uint32_t> &allTensorIds, std::set<uint32_t> &allOutTensorIds);

    //! \brief 获得当前图中所有的tensor id以及所有的输出tensor id
    //!
    //! \param outTensorIds 更新之前的所有输出张量id
    //! \param allOutTensorIdsNew 更新之后的所有输出张量id
    //!
    void updateAllTensorIDs(const std::set<uint32_t> &outTensorIds, const std::set<uint32_t> &allOutTensorIdsNew);
    //! \brief 获得算子的tiling key
    //!
    //! \param fusionClass 融合类型
    //! \return 字符串类型的tiling key
    //!
    std::string getTilingKey(const std::string &fusionClass);

    //! \brief 配置待创建的operation类型
    //!
    //! \param param 待创建的operation参数
    //! \param fusionClass 融合类型
    //!
    void SetFusionParam(atb::infer::FusionParam &param, const std::string &fusionClass);

    //! \brief 通过用户的指定来打开特定的融合类型
    //!
    //! \param fusionClassArray 用户指定的融合类型
    //!
    void SetFusionClass(const std::set<std::string> &fusionClassArray);

    //! \brief 存储BiShengIR的home路径构造的隐藏文件件
    //!
    //! 用来存储BiShnegIR产生的二进制
    //!
    std::string homePath_;

    //! \brief 待被分析的ATB图
    //!
    atb::GraphParam &graph_;

    //! \brief 存储图中可以被融合的Linear matmul nodes
    //!
    std::set<uint32_t> linearNodes_;

    //! \brief 存储图中可以被融合的add nodes
    //!
    std::set<uint32_t> eleAddNodes_;

    //! \brief 存储图中可以被融合的激活且类型为gelu的matmul-gelu nodes
    //!
    std::set<uint32_t> actGeluNodes_;

    //! \brief 存储图中可以被融合的激活且类型为sigmoid的matmul-sigmoid nodes
    //!
    std::set<uint32_t> actSigmoidNodes_;

    //! \brief 存储图中可以被融合的激活且类型为swiglu的matmul-swiglu nodes
    //!
    std::set<uint32_t> actSwiGluNodes_;

    //! \brief 用户指定的融合类型
    //!
    std::set<std::string> fusionClassMap_;
};
} // namespace atb
#endif