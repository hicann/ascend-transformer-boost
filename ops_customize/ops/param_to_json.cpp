/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <string>
#include <vector>
#include <sstream>
#include <nlohmann/json.hpp>
#include "atbops/params/params.h"
#include "mki/utils/SVector/SVector.h"
#include "mki/utils/stringify/stringify.h"
#include "mki/utils/any/any.h"
#include "mki/utils/log/log.h"

using namespace Mki;

namespace AtbOps {
template <typename T> std::vector<T> SVectorToVector(const SVector<T> &svector)
{
    std::vector<T> tmpVec;
    tmpVec.resize(svector.size());
    for (size_t i = 0; i < svector.size(); i++) {
        tmpVec.at(i) = svector.at(i);
    }
    return tmpVec;
};

std::string BlockCopyToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::CustomizeBlockCopy specificParam = AnyCast<OpParam::CustomizeBlockCopy>(param);
    paramsJson["type"] = specificParam.type;
    return paramsJson.dump();
}


REG_STRINGIFY(OpParam::CustomizeBlockCopy, BlockCopyToJson);
} // namespace AtbOps
