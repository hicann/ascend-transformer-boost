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

#include "apply_rotary_emb_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
ApplyRotaryEmbRunner::ApplyRotaryEmbRunner(const ApplayRotaryEmbParam &param) : OpsRunner("ApplyRotaryEmbRunner", RUNNER_TYPE_APPLY_ROTARY_EMB), param_(param)
{
    
    ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner called";
    
    kernelGraph_.inTensors.resize(3);
    AsdOps::Tensor &xq = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &xk = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &freqs_cis = kernelGraph_.inTensors.at(2);
    kernelGraph_.outTensors.resize(2);
    AsdOps::Tensor &xq_ = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &xk_ = kernelGraph_.outTensors.at(1);
    
    kernelGraph_.internalTensors.resize(21);
    size_t internalParamIndex = 0;

    AsdOps::Tensor &xq_real = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &xq_imga = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &xk_real = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &xk_imga = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &freqs_float16 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &freqs_real = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &freqs_imga = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &mulTmp1 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &mulTmp2 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &mulTmp3 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &mulTmp4 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &real1 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &imga1 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &xqConcatfloat16 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &mulTmp5 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &mulTmp6 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &mulTmp7 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &mulTmp8 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &real2 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &imga2 = kernelGraph_.internalTensors.at(internalParamIndex++);
    AsdOps::Tensor &xkConcatfloat16 = kernelGraph_.internalTensors.at(internalParamIndex++);


    size_t nodeId = 0;
    size_t nodeCount = 20;
    kernelGraph_.nodes.resize(nodeCount);
    auto &xqSplitNode = kernelGraph_.nodes[nodeId++];
    auto &xkSplitNode = kernelGraph_.nodes[nodeId++];
    auto &freqsCastNode = kernelGraph_.nodes[nodeId++];
    auto &freqsSplitNode = kernelGraph_.nodes[nodeId++];


    auto &mul1Node = kernelGraph_.nodes[nodeId++];
    auto &mul2Node = kernelGraph_.nodes[nodeId++];
    auto &mul3Node = kernelGraph_.nodes[nodeId++];
    auto &mul4Node = kernelGraph_.nodes[nodeId++];
    auto &sub1Node = kernelGraph_.nodes[nodeId++];
    auto &add1Node = kernelGraph_.nodes[nodeId++];
    auto &concat1Node = kernelGraph_.nodes[nodeId++];
    auto &reshapeXqOutNode = kernelGraph_.nodes[nodeId++];

    auto &mul5Node = kernelGraph_.nodes[nodeId++];
    auto &mul6Node = kernelGraph_.nodes[nodeId++];
    auto &mul7Node = kernelGraph_.nodes[nodeId++];
    auto &mul8Node = kernelGraph_.nodes[nodeId++];
    auto &sub2Node = kernelGraph_.nodes[nodeId++];
    auto &add2Node = kernelGraph_.nodes[nodeId++];
    auto &concat2Node = kernelGraph_.nodes[nodeId++];
    auto &reshapeXkOutNode = kernelGraph_.nodes[nodeId++];

    ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner start node";

    xqSplitNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{4, 2}};
    xqSplitNode.inTensors = {&xq};
    xqSplitNode.outTensors = {&xq_real, &xq_imga};
    xqSplitNode.inTensorViewFuncs.resize(xqSplitNode.inTensors.size());
    xqSplitNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner  xq slip to real and imga";
        newDims = {oldDims.at(0) , oldDims.at(1), oldDims.at(2),oldDims.at(3)/2, 2};
    };
    xqSplitNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    xkSplitNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{4, 2}};
    xkSplitNode.inTensors = {&xk};
    xkSplitNode.outTensors = {&xk_real, &xk_imga};
    xkSplitNode.inTensorViewFuncs.resize(xkSplitNode.inTensors.size());
    xkSplitNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner  xk slip to real and imga";
        newDims = {oldDims.at(0) , oldDims.at(1), oldDims.at(2),oldDims.at(3)/2, 2};
    };
    xkSplitNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    freqsCastNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    freqsCastNode.inTensors = {&freqs_cis};
    freqsCastNode.outTensors = {&freqs_float16};

    freqsSplitNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{4, 2}};
    freqsSplitNode.inTensors = {&freqs_float16};
    freqsSplitNode.outTensors = {&freqs_real, &freqs_imga};

    freqsSplitNode.inTensorViewFuncs.resize(freqsSplitNode.inTensors.size());
    freqsSplitNode.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner  freqs change shape to 1 "<< xq.desc.dims.at(1) <<" 1 "<<xq.desc.dims.at(3)/2 <<" 2";
        newDims = {1 , xq.desc.dims.at(1), 1, xq.desc.dims.at(3)/2 , 2};
    };
    freqsSplitNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner start mul node";

    mul1Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mul1Node.inTensors = {&xq_real, &freqs_real};
    mul1Node.outTensors = {&mulTmp1};
    mul1Node.inTensorViewFuncs.resize(mul1Node.inTensors.size());
    mul1Node.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner  xq_real * freqs_cis_real xq_real"  ;
        newDims = oldDims;
    };

    mul2Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mul2Node.inTensors = {&xq_imga, &freqs_imga};
    mul2Node.outTensors = {&mulTmp2};
    mul2Node.inTensorViewFuncs.resize(mul2Node.inTensors.size());
    mul2Node.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner  xq_imga * freqs_cis_imga xq_imga" << xq_real.data ;
        newDims = oldDims;
    };

    mul3Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mul3Node.inTensors = {&xq_imga, &freqs_real};
    mul3Node.outTensors = {&mulTmp3};

    mul4Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mul4Node.inTensors = {&xq_real, &freqs_imga};
    mul4Node.outTensors = {&mulTmp4};

    sub1Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_SUB})};
    sub1Node.inTensors = {&mulTmp1, &mulTmp2};
    sub1Node.outTensors = {&real1};
    sub1Node.inTensorViewFuncs.resize(sub1Node.inTensors.size());
    sub1Node.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner  tmp1-tmp2  tmp1" << mulTmp1.data <<"\r\n tmp2 "<<mulTmp2.data  ;
        newDims = oldDims;
    };

    add1Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    add1Node.inTensors = {&mulTmp3, &mulTmp4};
    add1Node.outTensors = {&imga1};

    concat1Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({4})};
    concat1Node.inTensors = {&real1, &imga1};
    concat1Node.outTensors = {&xqConcatfloat16};
    
	AsdOps::OpParam::Transpose permuteContextNodeParam =
        {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0,1,2,3}};
    reshapeXqOutNode.opDesc = {0, "TransposeOperation", permuteContextNodeParam};
    reshapeXqOutNode.inTensors = {&xqConcatfloat16};
    reshapeXqOutNode.outTensors = {&xq_};
    reshapeXqOutNode.inTensorViewFuncs.resize(reshapeXqOutNode.inTensors.size());
    reshapeXqOutNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = xq.desc.dims;
    };

    mul5Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mul5Node.inTensors = {&xk_real, &freqs_real};
    mul5Node.outTensors = {&mulTmp5};

    mul6Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mul6Node.inTensors = {&xk_imga, &freqs_imga};
    mul6Node.outTensors = {&mulTmp6};

    mul7Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mul7Node.inTensors = {&xk_imga, &freqs_real};
    mul7Node.outTensors = {&mulTmp7};

    
    mul8Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mul8Node.inTensors = {&xk_real, &freqs_imga};
    mul8Node.outTensors = {&mulTmp8};


    sub2Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_SUB})};
    sub2Node.inTensors = {&mulTmp5, &mulTmp6};
    sub2Node.outTensors = {&real2};

    add2Node.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    add2Node.inTensors = {&mulTmp7, &mulTmp8};
    add2Node.outTensors = {&imga2};

    concat2Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({4})};
    concat2Node.inTensors = {&real2, &imga2};
    concat2Node.outTensors = {&xkConcatfloat16};
    
	reshapeXkOutNode.opDesc = {0, "TransposeOperation", permuteContextNodeParam};
    reshapeXkOutNode.inTensors = {&xkConcatfloat16};
    reshapeXkOutNode.outTensors = {&xk_};
    reshapeXkOutNode.inTensorViewFuncs.resize(reshapeXqOutNode.inTensors.size());
    reshapeXkOutNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = xk.desc.dims;
    };
    
    ASD_LOG(INFO) << "ApplyRotaryEmbRunner::ApplyRotaryEmbRunner call end";
}

ApplyRotaryEmbRunner::~ApplyRotaryEmbRunner() {}
} // namespace AclTransformer
