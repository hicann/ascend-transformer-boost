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
#ifndef ATB_SPEED_MODELS_CHATGML6B_ENCODER_ROPE_MODEL_H
#define ATB_SPEED_MODELS_CHATGML6B_ENCODER_ROPE_MODEL_H
#include <atb/svector.h>
#include "atb_speed/base/model.h"

namespace atb_speed {
class ChatGlm6BEncoderRopeModel : public Model {
public:
    struct Param {
        double layerNormEps = 0;
        int headNum = 0;
        bool transKey = false;
        int dk = 0;
        int layerNum = 0;
        float residualAddScale = 0;
        int beginNormAxis = 1;
        void FromString(const std::string &param);
    };

    ChatGlm6BEncoderRopeModel(const std::string &param);
    ~ChatGlm6BEncoderRopeModel();
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph() override;

private:
    Param param_;
};
} // namespace atb_speed
#endif
