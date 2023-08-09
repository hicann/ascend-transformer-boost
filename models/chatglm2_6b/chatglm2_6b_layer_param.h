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
#ifndef OPS_CHATGML2_6B_LAYER_PARAM_H
#define OPS_CHATGML2_6B_LAYER_PARAM_H

namespace AclTransformer {
struct ChatGlm2LayerParam {
    int64_t numHeadsPerPartition;
    int64_t numGroupsPerPartition;
    int64_t hiddenSizePerHead;
    int64_t layerId;
    float rmsNormEps = 0;
    float residualAddScale = 0;
    float preScale = 0;
    float postScale = 0;
    bool transKey = false;
    std::string model = "chatglm2_6b";
};
} // namespace AclTransformer
#endif