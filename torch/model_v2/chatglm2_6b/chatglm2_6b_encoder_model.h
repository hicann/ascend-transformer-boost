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
#ifndef CHATGML2_6B_ENCODER_MODEL_H
#define CHATGML2_6B_ENCODER_MODEL_H
#include "torch/model_v2/model.h"

namespace AclTransformer {
class ChatGlm2EncoderModel : public Model {
public:
    struct Param {
        bool transKey = false;
        int layerNum = 0;
        float residualAddScale = 0;
        float rmsNormEps = 0;
        int numHeadsPerPartition = 0;
        int hiddenSizePerHead = 0;
        int numGroupsPerPartition = 0;
        void FromString(const std::string &param);
    };

    ChatGlm2EncoderModel(const std::string &param);
    ~ChatGlm2EncoderModel();
    uint64_t GetInTensorCount() const override;
    uint64_t GetOutTensorCount() const override;
    AsdOps::Status InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                              std::vector<AsdOps::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph() override;

private:
    Param param_;
};
} // namespace AclTransformer
#endif
