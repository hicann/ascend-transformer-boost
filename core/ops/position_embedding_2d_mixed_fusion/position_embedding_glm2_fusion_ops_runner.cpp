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
#include "position_embedding_fusion_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/rope.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingGlm2FusionOpsRunner::PositionEmbeddingGlm2FusionOpsRunner(const PositionEmbeddingFusionParam &param)
    : OpsRunner("PositionEmbeddingGlm2FusionOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_2D_MIXED_FUSION), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingGlm2FusionOpsRunner::PositionEmbeddingGlm2FusionOpsRunner called, headNum: "
                  << param_.headNum;
    const size_t inTensorSize = 5;
    const size_t outTensorSize = 3;
    const size_t interTensorSize = 10;
    const size_t nodeSize = 10;
    const int32_t kqvSliceSize = 3;
    kernelGraph_.inTensors.resize(inTensorSize);
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &ropeCache = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(2);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(index2);  // V

    kernelGraph_.internalTensors.resize(interTensorSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &valueIntermediate = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &ropeCacheSlice = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &asStrided0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided1Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided2Node = kernelGraph_.nodes[nodeNum++];
    auto &muls0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided3Node = kernelGraph_.nodes[nodeNum++];
    auto &split0Node = kernelGraph_.nodes[nodeNum++];
    auto &rope0Node = kernelGraph_.nodes[nodeNum++];

    int64_t qLayerDim = param_.numHeadPerPartition * param_.hiddenSizePerHead;
    int64_t kLayerDim = param_.numGroupsPerPartition * param_.hiddenSizePerHead;
    int64_t np = param_.numHeadPerPartition;
    int64_t hn = param_.hiddenSizePerHead;
    int64_t gp = param_.numGroupsPerPartition;

    // split qkv
    asStrided0Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided0Node.inTensors = {&mixedQkv};
    asStrided0Node.outTensors = {&qLayer};
    asStrided0Node.inferShapePreFunc = [qLayerDim](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split q";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), qLayerDim};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2), dims.at(2), 1};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    asStrided1Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided1Node.inTensors = {&mixedQkv};
    asStrided1Node.outTensors = {&kLayer};
    asStrided1Node.inferShapePreFunc = [qLayerDim, kLayerDime](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split k";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), kLayerDime};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2), dims.at(2), 1};
        AsdOps::SVector<int64_t> offset = {qLayerDim};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    asStrided2Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided2Node.inTensors = {&mixedQkv};
    asStrided2Node.outTensors = {&valueIntermediate};
    asStrided2Node.inferShapePreFunc = [qLayerDim, kLayerDime](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split v
        ";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), kLayerDime};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2), dims.at(2), 1};
        AsdOps::SVector<int64_t> offset = {qLayerDim + kLayerDime};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    // view qkv before slice
    muls0Node.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, 1}};
    muls0Node.inTensors = {&valueIntermediate};
    muls0Node.outTensors = {&value};
    muls0Node.inTensorViewFuncs.resize(muls0Node.inTensors.size());
    muls0Node.inTensorViewFuncs.at(0) = [gp, hn](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), gp, hn};
    };

    InferShapePreFunc split1InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 2}});
    };
    
    // slice rope with sq
    asStrided3Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided3Node.inTensors = {&ropeCache};
    asStrided3Node.outTensors = {&ropeCacheSlice};
    asStrided3Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t sq_len = kernelGraph_.inTensors.at(0).desc.dims[0];
        ASD_LOG(INFO) << "slice rope with sq"<< sq_len;
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {sq_len, dims.at(1), dims.at(2), dims.at(3)};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3), 1};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    // split to first half (cos) and second half (sin)
    split0Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split0Node.inTensors = {&ropeCacheSlice};
    split0Node.outTensors = {&cos, &sin};
    split0Node.inferShapePreFunc = split1InferShape;

    ViewFunc ropeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims.resize(2);
        newDims.at(0) = oldDims.at(0);
        newDims.at(1) = oldDims.at(1); 
    }; 

    ViewFunc ropeKqView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    const size_t ropeViewSize = 4;
    rope0Node.opDesc = {0, "RopeOperation", AsdOps::OpParam::Rope{AsdOps::OpParam::Rope::ROPEND, 2}};
    rope0Node.inTensors = {&qLayer, &kLayer, &cos, &sin, &seqLen};
    rope0Node.outTensors = {&qEmbedded, &kEmbedded};
    rope0Node.inTensorViewFuncs.resize(ropeViewSize);
    rope0Node.inTensorViewFuncs.at(0) = ropeKqView;
    rope0Node.inTensorViewFuncs.at(1) = ropeKqView;
    rope0Node.inTensorViewFuncs.at(2) = ropeCosSinView;
    rope0Node.inTensorViewFuncs.at(3) = ropeCosSinView;
    rope0Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        runInfo.GetInTensor(4).desc.dtype = AsdOps::TENSOR_DTYPE_UINT32;
    };
}

PositionEmbeddingGlm2FusionOpsRunner::~PositionEmbeddingGlm2FusionOpsRunner() {}
} // namespace AclTransformer
