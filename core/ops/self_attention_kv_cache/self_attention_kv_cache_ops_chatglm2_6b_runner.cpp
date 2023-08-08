#include "self_attention_kv_cache_ops_chatglm2_6b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheOpsChatGlm26bRunner::SelfAttentionKvCacheOpsChatGlm26bRunner(const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsChatGlm26bRunner", RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKvCacheOpsChatGlm26bRunner::SelfAttentionKvCacheOpsChatGlm26bRunner called"
                  << "transKey: " << param_.transKey << ",dk: " << param_.dk << ",headNum: " << param_.headNum
                  << ",layerId: " << param_.layerId << ", preScale: " << param_.preScale << ", postscale" << param_.postScale
                  << ", numAttentionHeadsPerPartition" << param_.numAttentionHeadsPerPartition
                  << ", hiddenSizePerAttentionHead " << param_.hiddenSizePerAttentionHead
                  << ", numMultiQueryGroupsPerPartition" << param_.numMultiQueryGroupsPerPartition << ", model " << param_.model;
    kernelGraph_.inTensors.resize(5);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastKey = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastValue = kernelGraph_.inTensors.at(inTensorNum++);

    const int outTensorSize = 3;
    kernelGraph_.outTensors.resize(outTensorSize);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(outTensorNum++);

    const int internalTensorSize = 10;
    kernelGraph_.internalTensors.resize(internalTensorSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVout = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &presentKeyExpand = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &presentValueExpand = kernelGraph_.internalTensors.at(internalTensorNum++);

    const int nodeSize = 13;
    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &mulsQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &catKeyNode = kernelGraph_.nodes.at(nodeNum++);
    auto &expandNode0 = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmQkNode = kernelGraph_.nodes.at(nodeNum++);
    auto &mulsMaskOutNode = kernelGraph_.nodes.at(nodeNum++);
    auto &softMaxNode = kernelGraph_.nodes.at(nodeNum++);
    auto &catValueNode = kernelGraph_.nodes.at(nodeNum++);
    auto &expandNode1 = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteContextNode = kernelGraph_.nodes.at(nodeNum++);


    float varAttr = 1.0 / (sqrt(param_.dk) * (param_.preScale));
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulsQNode.inTensors = {&mixedQuery};
    mulsQNode.outTensors = {&divOut};
    mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    AsdOps::OpParam::Transpose permuteQNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
    permuteQNode.opDesc = {0, "TransposeOperation", permuteQNodeParam};
    permuteQNode.inTensors = {&divOut};
    permuteQNode.outTensors = {&transposedQ};
    permuteQNode.inTensorViewFuncs.resize(permuteQNode.inTensors.size());
    permuteQNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };

    catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    catKeyNode.inTensors = {&pastKey, &mixedKey};
    catKeyNode.outTensors = {&presentKey};
    catKeyNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    int64_t np = param_.numAttentionHeadsPerPartition;
    int64_t hn = param_.hiddenSizePerAttentionHead;
    int64_t gp = param_.numMultiQueryGroupsPerPartition;
    InferShapePreFunc expandInferShape = [np, gp](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), np / gp, dims.at(4)};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(4), dims.at(2) * dims.at(4), dims.at(4), 0, 1};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };
    expandNode0.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    expandNode0.inTensors = {&presentKey};
    expandNode0.outTensors = {&presentKeyExpand};
    expandNode0.inTensorViewFuncs.resize(expandNode0.inTensors.size());
    expandNode0.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2), 1, oldDims.at(3)};
    };
    expandNode0.inferShapePreFunc = expandInferShape;

    AsdOps::OpParam::Transpose permuteKNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 2, 0}};
    permuteKNode.opDesc = {0, "TransposeOperation", permuteKNodeParam};
    permuteKNode.inTensors = {&presentKeyExpand};
    permuteKNode.outTensors = {&transposedK};
    permuteKNode.inTensorViewFuncs.resize(permuteKNode.inTensors.size());
    permuteKNode.inTensorViewFuncs[0] = [np, hn](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * np, hn};
    };

    bmmQkNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmQkNode.inTensors = {&transposedQ, &transposedK};
    bmmQkNode.outTensors = {&bmmQkOut};
    bmmQkNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    float postScale = param_.postScale;
    mulsMaskOutNode.opDesc = {0, "ElewiseOperation",
                            AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, postScale})};
    mulsMaskOutNode.inTensors = {&bmmQkOut};
    mulsMaskOutNode.outTensors = {&attentionScores};
    mulsMaskOutNode.inTensorViewFuncs.resize(mulsMaskOutNode.inTensors.size());
    mulsMaskOutNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    softMaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMaxNode.inTensors = {&attentionScores};
    softMaxNode.outTensors = {&attentionProbs};

    catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    catValueNode.inTensors = {&pastValue, &mixedValue};
    catValueNode.outTensors = {&presentValue};
    catValueNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    expandNode1.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    expandNode1.inTensors = {&presentValue};
    expandNode1.outTensors = {&presentValueExpand};
    expandNode1.inTensorViewFuncs.resize(expandNode1.inTensors.size());
    expandNode1.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2), 1, oldDims.at(3)};
    };
    expandNode1.inferShapePreFunc = expandInferShape;

    AsdOps::OpParam::Transpose permuteVNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
    permuteVNode.opDesc = {0, "TransposeOperation", permuteVNodeParam};
    permuteVNode.inTensors = {&presentValueExpand};
    permuteVNode.outTensors = {&transposedV};
    permuteVNode.inTensorViewFuncs.resize(permuteVNode.inTensors.size());
    permuteVNode.inTensorViewFuncs[0] = [np, hn](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * np, hn};
    };

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbs, &transposedV};
    bmmVNode.outTensors = {&bmmVout};
    bmmVNode.inTensorViewFuncs.resize(bmmVNode.inTensors.size());
    bmmVNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };
    bmmVNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    AsdOps::OpParam::Transpose permuteContextNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {2, 0, 1, 3}};
    permuteContextNode.opDesc = {0, "TransposeOperation", permuteContextNodeParam};
    permuteContextNode.inTensors = {&bmmVout};
    permuteContextNode.outTensors = {&context};
    permuteContextNode.inTensorViewFuncs.resize(permuteContextNode.inTensors.size());
    permuteContextNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };
}

SelfAttentionKvCacheOpsChatGlm26bRunner::~SelfAttentionKvCacheOpsChatGlm26bRunner() {}
} // namespace AclTransformer