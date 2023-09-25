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
#ifndef OPS_LLAMA13B_LLAMA13BLAYER_FUSION_QUANT_PARAM_H
#define OPS_LLAMA13B_LLAMA13BLAYER_FUSION_QUANT_PARAM_H

namespace AclTransformer {
struct LLaMA13BLayerFusionQuantParam {
    std::string model = "llama13b";
    float inputScale_1 = 1;
    int inputOffset_1 = 0;
    bool transposeA = false;
    bool transposeB = false;
    int headNum = 0;
    int dk = 0;
    float inputScale_2 = 1;
    int inputOffset_2 = 0;
    float scale = 1.0;
    float inputScale_3 = 1;
    int inputOffset_3 = 0;
    float inputScale_4 = 1;
    int inputOffset_4 = 0;
    int layerId = 0;
    AsdOps::SVector<int> seqLen;
    AsdOps::SVector<int> tokenOffset;
    int rotaryCoeff = 2;
    float rmsNormEps = 1e-12;
};
} // namespace AclTransformer
#endif