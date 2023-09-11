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
#ifndef ATB_SPEED_MODELS_CHATGLM130BLAYER_DECODER_FUSION_OPERATION_H
#define ATB_SPEED_MODELS_CHATGLM130BLAYER_DECODER_FUSION_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
enum Chatglm130BLayerDecoderFlashAttentionTensorId {
    IN_HIDDENSTATES_ID = 0,
    IN_NORMWEIGHT_ID,
    IN_NORMBIAS_ID,
    IN_QKVMIXEDWEIGHT_ID,
    IN_QKVMIXEDBIAS_ID,
    IN_SELFOUTLINEARWEIGHT_ID,
    IN_SELFOUTLINEARBIAS_ID,
    IN_SELFOUTNORMWEIGHT_ID,
    IN_SELFOUTNORMBIAS_ID,
    IN_MLPLINEARWEIGHT_ID,
    IN_MLPLINEARBIAS_ID,
    IN_MLPOUTLINEARWEIGHT_ID,
    IN_MLPOUTLINEARBIAS_ID,
    IN_POSITIONIDS_ID,
    IN_COS_ID,
    IN_SIN_ID,
    IN_ATTENTIONMASK_ID,
    IN_CACHEK_ID,
    IN_CACHEV_ID,
    IN_SEQLEN_ID,
    IN_TOKENOFFSET_ID,
    IN_LAYERID_ID,
    OUT_LAYEROUT_ID,
    INTERMEDIATE_INPUTNORMOUT_ID,
    INTERMEDIATE_MIXEDLINEAROUTQKV_ID,
    INTERMEDIATE_POSITIONEMBEDQ_ID,
    INTERMEDIATE_POSITIONEMBEDK_ID,
    INTERMEDIATE_VALUE_ID,
    INTERMEDIATE_SELFOUT_ID,
    INTERMEDIATE_SELFLINEAROUT_ID,
    INTERMEDIATE_SELFRESIDUALADDOUT_ID,
    INTERMEDIATE_SELFNORMOUT_ID,
    INTERMEDIATE_MLPOUT,
    INTERMEDIATE_MLPLINEAROUT_ID,
};

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


atb::Status CreateGlm130BLayerDecoderFusionOperation(const Glm130BLayerParam &param, atb::Operation **operation);

class ChatGlm130BLayerDecoderFlashAttentionBinder : public HostTensorBinder {
public:
    ChatGlm130BLayerDecoderFlashAttentionBinder();
    virtual ~ChatGlm130BLayerDecoderFlashAttentionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> tokenOffset_;
    atb::SVector<int64_t> seqLen_;
    int32_t layerId_ = 0;
};
} // namespace atb_speed
#endif