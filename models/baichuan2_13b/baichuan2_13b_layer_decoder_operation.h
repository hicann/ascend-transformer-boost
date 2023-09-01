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
#ifndef OPS_BAICHUAN2_13B_LAYER_DECODER_OPERATION_H
#define OPS_BAICHUAN2_13B_LAYER_DECODER_OPERATION_H

#include "acltransformer/graph_operation.h"
#include "baichuan2_13b_layer_param.h"

namespace AclTransformer {
class BaiChuan213BLayerDecoderOperation : public GraphOperation {
public:
    explicit BaiChuan213BLayerDecoderOperation(const BaiChuan213BLayerParam &param);
    ~BaiChuan213BLayerDecoderOperation();
    uint64_t GetInTensorCount() const override;
    uint64_t GetOutTensorCount() const override;

protected:
    AsdOps::Status InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const override;

private:
    BaiChuan213BLayerParam param_;
};
} // namespace AclTransformer
#endif