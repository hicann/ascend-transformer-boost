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
#ifndef LLAMA7B_DECODER_MODEL_H
#define LLAMA7B_DECODER_MODEL_H
#include "torch/model_v2/model.h"

namespace AclTransformer {
class Llama7BDecoderModel : public Model {
public:
    struct Param {
        double layerNormEps = 0;
        int headNum = 0;
        bool transKey = false;
        int dk = 0;
        int layerNum = 0;
        int rotaryCoeff = 2;
        AsdOps::SVector<int32_t> tokenOffset = {};
        AsdOps::SVector<int32_t> seqLen = {};
        void FromString(const std::string &param);
    };

    Llama7BDecoderModel(const std::string &param);
    ~Llama7BDecoderModel();
    uint64_t GetInTensorCount() const override;
    uint64_t GetOutTensorCount() const override;
    AsdOps::Status InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                              std::vector<AsdOps::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph() override;
    AsdOps::Status ParseVarintPackParam(const std::string &param, int nodeId, AsdOps::Any &variantPackParam) override;

private:
    Param param_;
};
} // namespace AclTransformer
#endif
