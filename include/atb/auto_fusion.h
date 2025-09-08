/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTO_FUSION_H
#define AUTO_FUSION_H
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <set>
#include "atb/context.h"
#include "atb/graph_op_builder.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#include "atb/operation.h"
#include "atb/svector.h"
#include "atb/types.h"
#include "atb/utils.h"

//!
//! \file auto_fusion.h
//!
//! \brief 定义算子自动融合AutoFusion类
//!
namespace atb {
//!
//! \class AutoFusion
//!
//! \brief 定义算子自动融合AutoFusion类
//!
//! 该接口类定义了算子自动融合工具类，提供了自动融合的函数方法。
//!
    
class AutoFusion {
public:
    //! \brief 默认构造函数
    //!
    //!
    AutoFusion() = default;
    //! \brief 构造函数
    //!
    //! \param graph AutoFusion类被创建的时候需要绑定的图结构
    //!
    AutoFusion(atb::GraphParam &graph)
    {
        (void)(graph);
    }

    //! \brief 析构函数。
    virtual ~AutoFusion() = default;

    //! \brief 执行auto fusion的入口函数
    //!
    //! \param fusionClassArray 用户指定的融合类型，默认是空，表示不触发算子自动融合
    //!
    virtual void DoAutoFusion(const std::set<std::string> &fusionClassArray = {}) = 0;
};

//! \brief auto fusion 工具的创建接口
//!
//! \param graph AutoFusion类被创建的时候需要绑定的图结构
//! 
//! \param autoFusionTool  auto fusion工具指针的handler
//!
//! \return Status类型的状态，表示创建auto fusion 工具是否成功
//!
Status CreateAutoFusionTool(atb::GraphParam &graph, atb::AutoFusion **autoFusionTool);
} // namespace atb
#endif