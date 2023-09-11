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
#ifndef ATB_SPEED_MODELS_CHATGLM130BLAYER_DECODER_OPERATION_H
#define ATB_SPEED_MODELS_CHATGLM130BLAYER_DECODER_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
struct Glm130BLayerParam {
    bool transKey = false;
    int headNum = 0;
    int dk = 0;
    int layerId = 0;
    int rank = 0;
    int rankSize = 1;
    float residualAddScale = 0;
    double layerNormEps = 0;
    std::string backend = "hccl";
    atb::SVector<int32_t> seqLen;
    atb::SVector<int32_t> tokenOffset;
};

atb::Status CreateGlm130BLayerDecoderFlashAttentionOperation(const Glm130BLayerParam &param, atb::Operation **operation);
} // namespace atb_speed
#endif
