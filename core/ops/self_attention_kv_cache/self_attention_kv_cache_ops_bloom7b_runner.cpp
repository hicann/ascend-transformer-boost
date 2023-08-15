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
#include "self_attention_kv_cache_ops_bloom7b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheOpsBloom7bRunner::SelfAttentionKvCacheOpsBloom7bRunner(const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsBloom7bRunner", RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param) {
    ASD_LOG(INFO) << "SelfAttentionKvCacheOpsBloom7bRunner::SelfAttentionKvCacheOpsBloom7bRunner called";
    const int inTensorSize = 10;
    kernelGraph_.inTensors.resize(inTensorSize);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &hiddenStates = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &linear1WeightTensor = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &linear1BiasTensor = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastK = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastV = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &alibi = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &linear2WeightTensor = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &linear2BiasTensor = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &attentionMask = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &residual = kernelGraph_.inTensors.at(inTensorNum++);

    const int outTensorSize = 3;
    kernelGraph_.outTensors.resize(outTensorSize);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentK = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentV = kernelGraph_.outTensors.at(outTensorNum++);

    const int internalTensorSize = 35;
    kernelGraph_.internalTensors.resize(internalTensorSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &transdata0ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdata1ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &FusedQKV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdatalinearWeight = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &matmul1ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdata2ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mixedQkv = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &value = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataPresentV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mmQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataMMQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulMMQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &maskFillOutFP32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &maskFillOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsFP32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdataAttentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &contextLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &matmul3ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdata6ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mixed2Qkv = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdatacontextLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedcontextLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdata26ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdata27ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &matmul28ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transdata29ResultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);

    const int nodeSize = 32;
    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &transdata0Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata1Node = kernelGraph_.nodes.at(nodeNum++);
    auto &matmul2Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata3Node = kernelGraph_.nodes.at(nodeNum++);
    auto &add4Node = kernelGraph_.nodes.at(nodeNum++);
    auto &split5Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeQ6Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeK7Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeV8Node = kernelGraph_.nodes.at(nodeNum++);
    auto &catKey9Node = kernelGraph_.nodes.at(nodeNum++);
    auto &catValue10Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQ11Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataK12Node = kernelGraph_.nodes.at(nodeNum++);
    auto &mmQk13Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataMMOut14Node = kernelGraph_.nodes.at(nodeNum++);
    auto &mulsMMOut15Node = kernelGraph_.nodes.at(nodeNum++);
    auto &addMMOut16Node = kernelGraph_.nodes.at(nodeNum++);
    auto &maskFill17Node = kernelGraph_.nodes.at(nodeNum++);
    auto &cast18Node = kernelGraph_.nodes.at(nodeNum++);
    auto &softMax19Node = kernelGraph_.nodes.at(nodeNum++);
    auto &cast20Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata21Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata22Node = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmContext23Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata24Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeQ25Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata26Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata27Node = kernelGraph_.nodes.at(nodeNum++);
    auto &matmul28Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdata29Node = kernelGraph_.nodes.at(nodeNum++);
    auto &add30Node = kernelGraph_.nodes.at(nodeNum++);
    auto &add31Node = kernelGraph_.nodes.at(nodeNum++);

    // first linear
    transdata0Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata0Node.inTensors = {&hiddenStates};
    transdata0Node.outTensors = {&transdata0ResultTensor};
    transdata0Node.inTensorViewFuncs.resize(transdata0Node.inTensors.size());
    transdata0Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(INFO) << TensorUtil::AsdOpsDimsToString(oldDims);
        oriDimA_ = oldDims;
        if (oldDims.size() == 2) {
            oriSize_ = 2;
            newDims = {1, oldDims.at(0), oldDims.at(1)};
        } else {
            newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
        }
    };

    transdata1Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata1Node.inTensors = {&linear1WeightTensor};
    transdata1Node.outTensors = {&transdata1ResultTensor};
    transdata1Node.inTensorViewFuncs.resize(transdata1Node.inTensors.size());
    transdata1Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimB_ = oldDims;
        newDims = {1, oldDims.at(0), oldDims.at(1)};
    };

    matmul2Node.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true, {0, 0}})};
    matmul2Node.inTensors = {&transdata0ResultTensor, &transdata1ResultTensor};
    matmul2Node.outTensors = {&matmul1ResultTensor};
    matmul2Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1, dim2;
        if (oriSize_ == 3) {
            dim0 = oriDimA_.at(0) * oriDimA_.at(1);
            dim1 = oriDimA_.at(2);
        } else {
            dim0 = oriDimA_.at(0);
            dim1 = oriDimA_.at(1);
        }
        if (true) {
            dim2 = oriDimB_.at(0);
        } else {
            dim2 = oriDimB_.at(1);
        }
        ASD_LOG(FATAL) << dim0 << " " << dim1 << " " << dim2;
        runInfo.SetOpDesc({0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true, {dim0, dim1, dim2}})});
    };

    ASD_LOG(INFO) << GetName() << " Transdata orgShape:[" << TensorUtil::AsdOpsDimsToString({0, 0}) << "]";
    transdata3Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata3Node.inTensors = {&matmul1ResultTensor};
    transdata3Node.outTensors = {&transdata2ResultTensor};
    transdata3Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1;
        if (oriSize_ == 3) {
            dim0 = oriDimA_.at(0) * oriDimA_.at(1);
        } else {
            dim0 = oriDimA_.at(0);
        }
        if (true) {
            dim1 = oriDimB_.at(0);
        } else {
            dim1 = oriDimB_.at(1);
        }
        runInfo.SetOpDesc({0,
            "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {dim0, dim1}})});
    };

    add4Node.opDesc = {
        0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    add4Node.inTensors = {&transdata2ResultTensor, &linear1BiasTensor};
    add4Node.outTensors = {&mixedQkv};

    // split
    split5Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{3, 3}};
    split5Node.inTensors = {&mixedQkv};
    split5Node.outTensors = {&qLayer, &kLayer, &value};
    split5Node.inTensorViewFuncs.resize(split5Node.inTensors.size());
    split5Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, 3, param_.dk};
    };

    // transpose*3
    AsdOps::OpParam::Transpose transposeQ6NodeParam = {
        AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeQ6Node.opDesc = {0, "TransposeOperation", transposeQ6NodeParam};
    transposeQ6Node.inTensors = {&qLayer};
    transposeQ6Node.outTensors = {&transposedQ};
    transposeQ6Node.inTensorViewFuncs.resize(transposeQ6Node.inTensors.size());
    transposeQ6Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, param_.dk};
    };

    AsdOps::OpParam::Transpose transposeK7NodeParam = {
        AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 3, 1}};
    transposeK7Node.opDesc = {0, "TransposeOperation", transposeK7NodeParam};
    transposeK7Node.inTensors = {&kLayer};
    transposeK7Node.outTensors = {&transposedK};
    transposeK7Node.inTensorViewFuncs.resize(transposeK7Node.inTensors.size());
    transposeK7Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, param_.dk};
    };

    AsdOps::OpParam::Transpose transposeV8NodeParam = {
        AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeV8Node.opDesc = {0, "TransposeOperation", transposeV8NodeParam};
    transposeV8Node.inTensors = {&value};
    transposeV8Node.outTensors = {&transposedV};
    transposeV8Node.inTensorViewFuncs.resize(transposeV8Node.inTensors.size());
    transposeV8Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, param_.dk};
    };

    // concat*2
    catKey9Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({2})};
    catKey9Node.inTensors = {&pastK, &transposedK};
    catKey9Node.outTensors = {&presentK};
    catKey9Node.inTensorViewFuncs.resize(catKey9Node.inTensors.size());
    catKey9Node.inTensorViewFuncs[1] = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * param_.headNum, param_.dk, oldDims.at(3)};
    };

    catValue10Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({1})};
    catValue10Node.inTensors = {&pastV, &transposedV};
    catValue10Node.outTensors = {&presentV};
    catValue10Node.inTensorViewFuncs.resize(catValue10Node.inTensors.size());
    catValue10Node.inTensorViewFuncs[1] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                              AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * param_.headNum, oldDims.at(2), param_.dk};
    };

    // baddbmm:transdata*2+mm+transdata+mul+add
    transdataQ11Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQ11Node.inTensors = {&transposedQ};
    transdataQ11Node.outTensors = {&transdataQ};
    transdataQ11Node.inTensorViewFuncs.resize(transdataQ11Node.inTensors.size());
    transdataQ11Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimC_ = oldDims;
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    transdataK12Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataK12Node.inTensors = {&presentK};
    transdataK12Node.outTensors = {&transdataK};
    transdataK12Node.inTensorViewFuncs.resize(transdataK12Node.inTensors.size());
    transdataK12Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimD_ = oldDims;
        newDims = oldDims;
    };

    mmQk13Node.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {0, 0}})};
    mmQk13Node.inTensors = {&transdataQ, &transdataK};
    mmQk13Node.outTensors = {&mmQkOut};
    mmQk13Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "MatMulOperation",
            AsdOps::OpParam::MatMul({false, false, {oriDimC_.at(2), oriDimC_.at(3), oriDimD_.at(2)}})});
    };

    transdataMMOut14Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataMMOut14Node.inTensors = {&mmQkOut};
    transdataMMOut14Node.outTensors = {&transdataMMQkOut};
    transdataMMOut14Node.inTensorViewFuncs.resize(transdataMMOut14Node.inTensors.size());
    transdataMMOut14Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) { newDims = oldDims; };
    transdataMMOut14Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "TransdataOperation",
            AsdOps::OpParam::Transdata(
                {AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {oriDimC_.at(2), oriDimD_.at(2)}})});
    };

    mulsMMOut15Node.opDesc = {0,
        "ElewiseOperation",
        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, param_.invNormFactorvarAttr})};
    mulsMMOut15Node.inTensors = {&transdataMMQkOut};
    mulsMMOut15Node.outTensors = {&mulMMQkOut};
    mulsMMOut15Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    addMMOut16Node.opDesc = {
        0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addMMOut16Node.inTensors = {&mulMMQkOut, &alibi};
    addMMOut16Node.outTensors = {&attentionScores};
    addMMOut16Node.inTensorViewFuncs.resize(addMMOut16Node.inTensors.size());
    addMMOut16Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(FATAL) << oldDims;
        newDims = oldDims;
    };
    addMMOut16Node.inTensorViewFuncs.at(
        1) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(FATAL) << oldDims;
        newDims = oldDims;
    };

    // masked_fill
    float maskValue = -65504.0;  //`torch.float16` has a minimum value of -65504.0
    maskFill17Node.opDesc = {0,
        "BroadcastOperation",
        AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MASKEDFILL, {maskValue}})};
    maskFill17Node.inTensors = {&attentionScores, &attentionMask};
    maskFill17Node.outTensors = {&maskFillOut};
    maskFill17Node.inTensorViewFuncs.resize(maskFill17Node.inTensors.size());
    maskFill17Node.inTensorViewFuncs.at(
        0) = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oriDimA_.at(0), param_.headNum, oriDimA_.at(1), oriDimD_.at(2)};
    };

    // cast fp16 to fp32
    cast18Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    cast18Node.inTensors = {&maskFillOut};
    cast18Node.outTensors = {&maskFillOutFP32};

    // softmax
    softMax19Node.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMax19Node.inTensors = {&maskFillOutFP32};
    softMax19Node.outTensors = {&attentionProbsFP32};

    // cast fp32 to fp16
    cast20Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    cast20Node.inTensors = {&attentionProbsFP32};
    cast20Node.outTensors = {&attentionProbs};

    // transdata*2+bmm+transdata
    transdata21Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata21Node.inTensors = {&attentionProbs};
    transdata21Node.outTensors = {&transdataAttentionProbs};
    transdata21Node.inTensorViewFuncs.resize(transdata21Node.inTensors.size());
    transdata21Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
        oriDimE_ = newDims;
    };

    transdata22Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata22Node.inTensors = {&presentV};
    transdata22Node.outTensors = {&transdataPresentV};
    transdata22Node.inTensorViewFuncs.resize(transdata22Node.inTensors.size());
    transdata22Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = oldDims;
        oriDimF_ = oldDims;
    };

    bmmContext23Node.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmContext23Node.inTensors = {&transdataAttentionProbs, &transdataPresentV};
    bmmContext23Node.outTensors = {&contextLayer};
    bmmContext23Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "MatMulOperation",
            AsdOps::OpParam::MatMul({false, false, {oriDimE_.at(1), oriDimE_.at(2), oriDimF_.at(2)}})});
    };

    transdata24Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata24Node.inTensors = {&contextLayer};
    transdata24Node.outTensors = {&transdatacontextLayer};
    transdata24Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "TransdataOperation",
            AsdOps::OpParam::Transdata(
                {AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {oriDimE_.at(1), oriDimF_.at(2)}})});
    };

    // transpose
    AsdOps::OpParam::Transpose transposeQ25NodeParam = {
        AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeQ25Node.opDesc = {0, "TransposeOperation", transposeQ25NodeParam};
    transposeQ25Node.inTensors = {&transdatacontextLayer};
    transposeQ25Node.outTensors = {&transposedcontextLayer};
    transposeQ25Node.inTensorViewFuncs.resize(transposeQ25Node.inTensors.size());
    transposeQ25Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(INFO) << TensorUtil::AsdOpsDimsToString(oldDims);
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), param_.dk};
    };

    // last linear
    transdata26Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata26Node.inTensors = {&transposedcontextLayer};
    transdata26Node.outTensors = {&transdata26ResultTensor};
    transdata26Node.inTensorViewFuncs.resize(transdata26Node.inTensors.size());
    transdata26Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {1, oldDims.at(1) * oldDims.at(0), oldDims.at(2) * oldDims.at(3)};
        oriDimI_ = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    transdata27Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata27Node.inTensors = {&linear2WeightTensor};
    transdata27Node.outTensors = {&transdata27ResultTensor};
    transdata27Node.inTensorViewFuncs.resize(transdata27Node.inTensors.size());
    transdata27Node.inTensorViewFuncs.at(
        0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriDimJ_ = oldDims;
        newDims = {1, oldDims.at(0), oldDims.at(1)};
    };

    matmul28Node.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true, {0, 0}})};
    matmul28Node.inTensors = {&transdata26ResultTensor, &transdata27ResultTensor};
    matmul28Node.outTensors = {&matmul28ResultTensor};
    matmul28Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0,
            "MatMulOperation",
            AsdOps::OpParam::MatMul({false, true, {oriDimI_.at(0) * oriDimI_.at(1), oriDimI_.at(2), oriDimJ_.at(0)}})});
    };

    transdata29Node.opDesc = {
        0, "TransdataOperation", AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata29Node.inTensors = {&matmul28ResultTensor};
    transdata29Node.outTensors = {&transdata29ResultTensor};
    transdata29Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1;
        dim0 = oriDimI_.at(0) * oriDimI_.at(1);
        dim1 = oriDimJ_.at(0);
        runInfo.SetOpDesc({0,
            "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {dim0, dim1}})});
    };

    add30Node.opDesc = {
        0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    add30Node.inTensors = {&transdata29ResultTensor, &linear2BiasTensor};
    add30Node.outTensors = {&mixed2Qkv};

    // last add
    add31Node.opDesc = {
        0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    add31Node.inTensors = {&mixed2Qkv, &residual};
    add31Node.outTensors = {&operationOutTensor};
}

SelfAttentionKvCacheOpsBloom7bRunner::~SelfAttentionKvCacheOpsBloom7bRunner() {
}
}  // namespace AclTransformer
