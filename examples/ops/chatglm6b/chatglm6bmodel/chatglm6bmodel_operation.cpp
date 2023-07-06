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
#include "chatglm6bmodel_operation.h"
#include "acltransformer/graph_operation.h"

namespace AclTransformer {

enum Chatglm6BModelTensorId {
    IN_HIDDENSTATES,
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTNORMBIAS,
    IN_FFNLINEARWEIGHT,
    IN_FFNLINEARBIAS,
    IN_FFNOUTLINEARWEIGHT,
    IN_FFNOUTLINEARBIAS,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    OUT_GLMMODELOUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_BASE,
};

ChatGlm6BModelOperation::ChatGlm6BModelOperation(const ChatGlm6BModelParam &param)
    : GraphOperation("ChatGlm6BModelOperation"), param_(param)
{
    uint64_t tensorId = 0;
    // in
    uint64_t hiddenStates = tensorId++;
    uint64_t positionIds = tensorId++;
    uint64_t cosTable = tensorId++;
    uint64_t sinTable = tensorId++;
    uint64_t attentionMask = tensorId++;

    uint64_t normWeight = tensorId++;
    uint64_t normBias = tensorId++;
    uint64_t qkvMixdWeight = tensorId++;
    uint64_t qkvMixdBias = tensorId++;
    uint64_t selfOutLinearWeight = tensorId++;
    uint64_t selfOutLinearBias = tensorId++;
    uint64_t selfOutNormWeight = tensorId++;
    uint64_t selfOutNormBias = tensorId++;
    uint64_t ffnLinearWeight = tensorId++;
    uint64_t ffnLinearBias = tensorId++;
    uint64_t ffnOutLinearWeight = tensorId++;
    uint64_t ffnOutLinearBias = tensorId++;
    tensorId = tensorId + 12 * (param_.layerNum - 1)
    uint64_t pastKey = tensorId++;
    tensorId = tensorId + (param_.layerNum - 1)
    uint64_t pastValue = tensorId++;
    tensorId = tensorId + (param_.layerNum - 1)
    // out
    uint64_t glmModelOut = tensorId++;
    uint64_t presentKey = tensorId++;
    tensorId = tensorId + (param_.layerNum - 1)
    uint64_t presentValue = tensorId++;
    tensorId = tensorId + (param_.layerNum - 1)
    // intermiate
    uint64_t intermidiateBase = tensorId++;

    opGraph_.inTensorSize = 5 + 14 * param_.layerNum;
    opGraph_.outTensorSize = 1 + 2 * param_.layerNum;
    opGraph_.intermediateTensorSize = param_.layerNum - 1;
    opGraph_.nodes.resize(param_.layerNum);

    for (size_t i = 0; i < param_.layerNum; i++) {
        GraphOperation::Node &chatGlm6bBlockNode = opGraph_.nodes.at(i);
        chatGlm6bBlockNode.reset(new AclTransformer::ChatGlm6BModelOperation({param_.layerNormEps,
                                                                          param_.headNum,
                                                                          param_.transKey,
                                                                          param_.dk,
                                                                          i,
                                                                          param_.residualAddScale}));

        sizt_t blockInHiddenStates = (i == 0) ? hiddenStates : (intermidiateBase + (i - 1));
        sizt_t blockOut = (i == (param_.layerNum-1)) ? glmModelOut : (intermidiateBase + i);
        size_t weightsOffset = (i - 1) * 12;
        size_t paskKeyOffset = i - 1;
        size_t pastValueOffset = i - 1;
        size_t presentKeyOffset = i - 1;
        size_t presentValueOffset = i - 1;

        chatGlm6bBlockNode.inTensorIds = {blockInHiddenStates, 
                                        normWeight + weightsOffset, normBias + weightsOffset,
                                        qkvMixdWeight + weightsOffset, qkvMixdBias + weightsOffset,
                                        selfOutLinearWeight + weightsOffset, selfOutLinearBias + weightsOffset,
                                        selfOutNormWeight + weightsOffset, selfOutNormBias + weightsOffset,
                                        ffnLinearWeight + weightsOffset, ffnLinearBias + weightsOffset,
                                        ffnOutLinearWeight + weightsOffset, ffnOutLinearBias + weightsOffset,
                                        positionIds, cosTable, sinTable,
                                        attentionMask, pastKey + paskKeyOffset, pastValue + pastValueOffset};
        inputNormNode.outTensorIds = {blockOut, presentKey + presentKeyOffset, presentValue + presentValueOffset};
    }
}

ChatGlm6BModelOperation::~ChatGlm6BModelOperation() {}

AsdOps::Status ChatGlm6BModelOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                       AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.resize(1 + param_.layerNum * 2);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    size_t int i = 1;
    for (; i < param_.layerNum, i++) {
        outTensorDescs.at(i) = inTensors.at(i + 4 + 12 * param_.layerNum).desc;
        outTensorDescs.at(i).dims.at(0) += 1;
    }
    for (; i < param_.layerNum, i++) {
        outTensorDescs.at(i) = inTensors.at(i + 4 + 13 * param_.layerNum).desc;
        outTensorDescs.at(i).dims.at(0) += 1;
    }
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer