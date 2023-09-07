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
#include "self_attention_ops_bloom7b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionOpsBloom7bRunner::SelfAttentionOpsBloom7bRunner(const SelfAttentionParam &param)
    : OpsRunner("SelfAttentionOpsBloom7bRunner", RUNNER_TYPE_SELF_ATTENTION), param_(param) {
    ASD_LOG(INFO) << "SelfAttentionOpsBloom7bRunner::SelfAttentionOpsBloom7bRunner called";
    const int inTensorSize = 3;
    kernelGraph_.inTensors.resize(inTensorSize);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &fusedQKV = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &alibi = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &attentionMask = kernelGraph_.inTensors.at(inTensorNum++);

    const int outTensorSize = 3;
    kernelGraph_.outTensors.resize(outTensorSize);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &transposedK = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &transposedV = kernelGraph_.outTensors.at(outTensorNum++);

    const int internalTensorSize = 18;
    kernelGraph_.internalTensors.resize(internalTensorSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &value = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdatatransposedV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mmQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataMMQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &maskFillOutFP32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &maskFillOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsFP32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataAttentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &contextLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdatacontextLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulMMQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);

    const int nodeSize = 19;
    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &split0Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeQ1Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeK2Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeV3Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQ4Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataK5Node = kernelGraph_.nodes.at(nodeNum++);
    auto &mmQk6Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataMMOut7Node = kernelGraph_.nodes.at(nodeNum++);
    auto &mulMMOut8Node = kernelGraph_.nodes.at(nodeNum++);
    auto &addMMOut9Node = kernelGraph_.nodes.at(nodeNum++);
    auto &maskFill10Node = kernelGraph_.nodes.at(nodeNum++);
    auto &cast11Node = kernelGraph_.nodes.at(nodeNum++);
    auto &softMax12Node = kernelGraph_.nodes.at(nodeNum++);
    auto &cast13Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata14Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata15Node = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmContext16Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata17Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeQ18Node = kernelGraph_.nodes.at(nodeNum++);

    // split
    split0Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{3, 3}};
    split0Node.inTensors = {&fusedQKV};
    split0Node.outTensors = {&qLayer, &kLayer, &value};
    split0Node.inTensorViewFuncs.resize(split0Node.inTensors.size());
    split0Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, 3, param_.dk};
    };

    // transpose*3
    AsdOps::OpParam::Transpose transposeQ1NodeParam = {
        AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeQ1Node.opDesc = {0, "TransposeOperation", transposeQ1NodeParam};
    transposeQ1Node.inTensors = {&qLayer};
    transposeQ1Node.outTensors = {&transposedQ};
    transposeQ1Node.inTensorViewFuncs.resize(transposeQ1Node.inTensors.size());
    transposeQ1Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, param_.dk};
    };

    AsdOps::OpParam::Transpose transposeK2NodeParam = {
        AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 3, 1}};
    transposeK2Node.opDesc = {0, "TransposeOperation", transposeK2NodeParam};
    transposeK2Node.inTensors = {&kLayer};
    transposeK2Node.outTensors = {&transposedK};
    transposeK2Node.inTensorViewFuncs.resize(transposeK2Node.inTensors.size());
    transposeK2Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, param_.dk};
    };

    AsdOps::OpParam::Transpose transposeV3NodeParam = {
        AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeV3Node.opDesc = {0, "TransposeOperation", transposeV3NodeParam};
    transposeV3Node.inTensors = {&value};
    transposeV3Node.outTensors = {&transposedV};
    transposeV3Node.inTensorViewFuncs.resize(transposeV3Node.inTensors.size());
    transposeV3Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, param_.dk};
    };

    // baddbmm:transdata*2+mm+transdata+mul+add
    transdataQ4Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQ4Node.inTensors = {&transposedQ};
    transdataQ4Node.outTensors = {&transdataQ};
    transdataQ4Node.inTensorViewFuncs.resize(transdataQ4Node.inTensors.size());
    transdataQ4Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimC_ = oldDims;
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    transdataK5Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataK5Node.inTensors = {&transposedK};
    transdataK5Node.outTensors = {&transdataK};
    transdataK5Node.inTensorViewFuncs.resize(transdataK5Node.inTensors.size());
    transdataK5Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimD_ = oldDims;
        newDims = oldDims;
    };

    mmQk6Node.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {0, 0}})};
    mmQk6Node.inTensors = {&transdataQ, &transdataK};
    mmQk6Node.outTensors = {&mmQkOut};
    mmQk6Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "MatMulOperation",
            AsdOps::OpParam::MatMul({false, false, {oriDimC_.at(2), oriDimC_.at(3), oriDimD_.at(2)}})});
    };

    transdataMMOut7Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataMMOut7Node.inTensors = {&mmQkOut};
    transdataMMOut7Node.outTensors = {&transdataMMQkOut};
    transdataMMOut7Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "TransdataOperation",
            AsdOps::OpParam::Transdata(
                {AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {oriDimC_.at(2), oriDimD_.at(2)}})});
    };

    float varAttr = 1.0f / std::sqrt(param_.dk);
    mulMMOut8Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulMMOut8Node.inTensors = {&transdataMMQkOut};
    mulMMOut8Node.outTensors = {&mulMMQkOut};

    addMMOut9Node.opDesc = {
        0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addMMOut9Node.inTensors = {&mulMMQkOut, &alibi};
    addMMOut9Node.outTensors = {&attentionScores};

    // masked_fill
    float maskValue = -65504.0;  //`torch.float16` has a minimum value of -65504.0
    maskFill10Node.opDesc = {0,
        "BroadcastOperation",
        AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MASKEDFILL, {maskValue}})};
    maskFill10Node.inTensors = {&attentionScores, &attentionMask};
    maskFill10Node.outTensors = {&maskFillOut};
    maskFill10Node.inTensorViewFuncs.resize(maskFill10Node.inTensors.size());
    maskFill10Node.inTensorViewFuncs.at(
        0) = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    // cast fp16 to fp32
    cast11Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    cast11Node.inTensors = {&maskFillOut};
    cast11Node.outTensors = {&maskFillOutFP32};

    // softmax
    softMax12Node.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMax12Node.inTensors = {&maskFillOutFP32};
    softMax12Node.outTensors = {&attentionProbsFP32};

    // cast fp32 to fp16
    cast13Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    cast13Node.inTensors = {&attentionProbsFP32};
    cast13Node.outTensors = {&attentionProbs};

    // transdata*2+bmm+transdata
    transdata14Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata14Node.inTensors = {&attentionProbs};
    transdata14Node.outTensors = {&transdataAttentionProbs};
    transdata14Node.inTensorViewFuncs.resize(transdata14Node.inTensors.size());
    transdata14Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
        oriDimE_ = newDims;
    };

    transdata15Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata15Node.inTensors = {&transposedV};
    transdata15Node.outTensors = {&transdatatransposedV};
    transdata15Node.inTensorViewFuncs.resize(transdata15Node.inTensors.size());
    transdata15Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = oldDims;
        oriDimF_ = oldDims;
    };

    bmmContext16Node.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmContext16Node.inTensors = {&transdataAttentionProbs, &transdatatransposedV};
    bmmContext16Node.outTensors = {&contextLayer};
    bmmContext16Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "MatMulOperation",
            AsdOps::OpParam::MatMul({false, false, {oriDimE_.at(1), oriDimE_.at(2), oriDimF_.at(2)}})});
    };

    transdata17Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata17Node.inTensors = {&contextLayer};
    transdata17Node.outTensors = {&transdatacontextLayer};
    transdata17Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "TransdataOperation",
            AsdOps::OpParam::Transdata(
                {AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {oriDimE_.at(1), oriDimF_.at(2)}})});
    };

    // transpose
    AsdOps::OpParam::Transpose transposeQ18NodeParam = {
        AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeQ18Node.opDesc = {0, "TransposeOperation", transposeQ18NodeParam};
    transposeQ18Node.inTensors = {&transdatacontextLayer};
    transposeQ18Node.outTensors = {&operationOutTensor};
    transposeQ18Node.inTensorViewFuncs.resize(transposeQ18Node.inTensors.size());
    transposeQ18Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), param_.dk};
    };
}

SelfAttentionOpsBloom7bRunner::~SelfAttentionOpsBloom7bRunner() {
}
}  // namespace AclTransformer
