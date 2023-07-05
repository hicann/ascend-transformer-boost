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
#include "acltransformer/variant_pack.h"
#include <sstream>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
std::string VariantPack::ToString() const
{
    std::stringstream ss;
    ss << "tilingBuffer:" << tilingBuffer << ", tilingBufferSize:" << tilingBufferSize << workspaceBuffer << ":"
       << workspaceBuffer << ", workspaceBufferSize:" << workspaceBufferSize
       << ", intermediateBuffer:" << intermediateBuffer << ", intermediateBufferSize:" << intermediateBufferSize
       << std::endl;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << TensorUtil::AsdOpsTensorToString(inTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << TensorUtil::AsdOpsTensorToString(outTensors.at(i)) << std::endl;
    }

    return ss.str();
}
} // namespace AclTransformer