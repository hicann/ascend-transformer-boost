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
#include "acltransformer/operation_graph.h"
#include <sstream>

namespace AclTransformer {
static std::string Join(const AsdOps::SVector<uint64_t> &ids)
{
    std::string ret;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i == 0) {
            ret.append(std::to_string(i));
        } else {
            ret.append(", " + std::to_string(i));
        }
    }
    return ret;
}

std::string OperationGraph::ToString() const
{
    std::stringstream ss;
    ss << "inTensorSize:" << inTensorSize << ", outTensorSize:" << outTensorSize
       << ", intermediateTensorSize:" << intermediateTensorSize;
    for (size_t i = 0; i < nodes.size(); ++i) {
        ss << "\nnode[" << i << "]: operation:" << nodes.at(i).operation << ", inTensorIds:["
           << Join(nodes.at(i).inTensorIds) << "], outTensorIds:[" << Join(nodes.at(i).outTensorIds) << "]";
    }
    return ss.str();
}

void OperationGraph::LayerSingle(OperationGraph &other) const {}
} // namespace AclTransformer
