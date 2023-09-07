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
#ifndef OPS_BLOOM7B_BLOOM7BLAYER_PARALLEL_PARAM_H
#define OPS_BLOOM7B_BLOOM7BLAYER_PARALLEL_PARAM_H

namespace AclTransformer {
struct Bloom7BLayerParallelParam {
    double layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    float invNormFactorvarAttr = 0;
    int activationFuncType = 1;
    int rank = 0;
    int rankSize = 1;
    std::string model = "bloom7b";
};
} // namespace AclTransformer
#endif