/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>
#include <iostream>

#include <acl/acl.h>

#include <atb/context.h>
#include <atb/operation.h>
#include <atb/infer_op_params.h>
#include <atb/types.h>

#include "context_utils.h"
#include "tensor_utils.h"

#include "models/base/param/param.h"
#include "models/base/param/layer_param.h"
#include "models/bloom/layer/bloom_decoder_layer.h"
static void Layer_Bloom_7B(benchmark::State &state)
{
    int round = 0;
    for (auto _ : state) {
        state.PauseTiming();
        aclInit(nullptr);
        uint32_t deviceId = 0;
        aclrtSetDevice(deviceId);
        aclrtStream stream;
        aclrtCreateStream(&stream);
        atb::Context *context = nullptr;
        atb::CreateContext(&context);
        context->SetExecuteStream(stream);
        atb::Operation *graphOperation = nullptr;
        {
            atb_speed::base::LayerParam layerParam;
            layerParam.layerId = 0;
            layerParam.numHiddenLayers = 1;
            layerParam.isFA = false;
            layerParam.isUnpadInputs = true;
            layerParam.isPrefill = false;
            layerParam.isBF16 = false;
            layerParam.isEdgeHardware = false;
            layerParam.enableSwiGLU = false;
            layerParam.enableLcoc = false;
            layerParam.enableMC2 = false;
            layerParam.enableSpeculate = false;
            layerParam.enableCompressHead = false;
            layerParam.enableOmniAttention = false;
            layerParam.useQKNorm = false;
            layerParam.enableSplitFuse = false;
            layerParam.enableLora = false;
            layerParam.enablePreFetchWeight = false;
            layerParam.loraEnableGMM = false;
            layerParam.enableKvQuant = false;
            layerParam.enableFA3 = false;
            layerParam.kvQuantHasOffset = true;
            layerParam.enableReduceQuant = false;
            layerParam.enableInterLayerAddNorm = false;
            layerParam.enableIntraLayerAddNorm = false;
            layerParam.enablePrefixCache = false;
            layerParam.attnBackend = atb_speed::common::OpBackend::ATB;
            layerParam.matmulBackend = atb_speed::common::OpBackend::ATB;
            layerParam.positionEmbeddingType = atb_speed::base::PositionEmbeddingType::ALIBI;
            layerParam.normEps = 1e-05;
            layerParam.normType = atb_speed::base::NormType::LAYER_NORM;
            layerParam.quantGroupSize = 0;
            layerParam.numAttentionHeadsPerRank = 32;
            layerParam.hiddenSizePerAttentionHead = 128;
            layerParam.numKeyValueHeadsPerRank = 32;
            layerParam.enableFlashComm = 0;
            layerParam.enableModelConfuscation = 0;
            layerParam.modelConfuscationFd = 0;
            layerParam.packQuantType = {atb_speed::common::PackQuantType::ALL_FP,
                                        atb_speed::common::PackQuantType::ALL_FP};
            layerParam.linearQuantType = {
                atb_speed::common::LinearType::FP,      atb_speed::common::LinearType::INVALID,
                atb_speed::common::LinearType::INVALID, atb_speed::common::LinearType::FP,
                atb_speed::common::LinearType::FP,      atb_speed::common::LinearType::INVALID,
                atb_speed::common::LinearType::FP};
            layerParam.linearTransposeType = {1, -1, -1, 1, 1, -1, 1};
            layerParam.linearHasBias = {true, true, true, true};
            layerParam.weightQuantType = "";
            layerParam.backend = "lccl";
            layerParam.tensorParallelInfo = {0, 1, "lccl", "", nullptr};
            layerParam.hasAttnTp = false;
            layerParam.attnTpRank = 0;
            layerParam.attnTpSize = 1;
            layerParam.attnTpDomain = "";
            layerParam.attnTpRankTableFile = "";
            layerParam.hasAttnDp = false;
            layerParam.attnDpRank = 0;
            layerParam.attnDpSize = 1;
            layerParam.attnDpDomain = "";
            layerParam.attnDpRankTableFile = "";
            layerParam.hasMlpTp = false;
            layerParam.mlpTpRank = 0;
            layerParam.mlpTpSize = 1;
            layerParam.mlpTpDomain = "";
            layerParam.mlpTpRankTableFile = "";
            layerParam.enableSwigluQuant = false;
            atb_speed::bloom::BloomDecoderLayer bloomDecoderLayer(layerParam);
            bloomDecoderLayer.BuildGraph(&graphOperation);
        }
        std::vector<atb::TensorDesc> inTensorDesc{
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {4096}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {4096}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {12288, 4096}, .dimNum = 2}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {12288}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {4096, 4096}, .dimNum = 2}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {4096}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {4096}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {4096}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {16384, 4096}, .dimNum = 2}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {16384}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {4096, 16384}, .dimNum = 2}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {4096}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1, 4096}, .dimNum = 2}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {32, 1, 4096}, .dimNum = 3}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {9, 128, 32, 128}, .dimNum = 4}},
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {9, 128, 32, 128}, .dimNum = 4}},
            {.dtype = ACL_INT32, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_INT32, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_INT32, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
            {.dtype = ACL_INT32, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1, 1}, .dimNum = 2}},
            {.dtype = ACL_INT32, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1}, .dimNum = 1}},
        };
        std::vector<std::string> filePath;
        for (int i = 0; i < 61; ++i) {
            filePath.push_back("/home/zouyanlong/msit_dump/tensors/0_15072/0/2_Prefill_layer/before/intensor" +
                               std::to_string(i) + ".bin");
        }
        std::vector<atb::Tensor> inTensor = FillTensorDataByFile(inTensorDesc, filePath);
        std::vector<atb::TensorDesc> outTensorDesc{
            {.dtype = ACL_FLOAT16, .format = ACL_FORMAT_ND, .shape = atb::Dims{.dims = {1, 4096}, .dimNum = 2}},
        };
        std::vector<atb::Tensor> outTensor = FillTensorDataByZero(outTensorDesc);
        atb::VariantPack variantPack;
        for (size_t i = 0; i < inTensor.size(); ++i) {
            variantPack.inTensors.push_back(inTensor[i]);
        }
        for (size_t i = 0; i < outTensor.size(); ++i) {
            variantPack.outTensors.push_back(outTensor[i]);
        }
        uint64_t workwpaceSize = 0;
        graphOperation->Setup(variantPack, workwpaceSize, context);
        void *workSpace = nullptr;
        aclrtMalloc(&workSpace, workwpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (round < 20)
            round++;
        else
            state.ResumeTiming();
        graphOperation->Execute(variantPack, (uint8_t *)workSpace, workwpaceSize, context);
        aclrtSynchronizeStream(stream);
        state.PauseTiming();
        atb::DestroyContext(context);
        atb::DestroyOperation(graphOperation);
        for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
            PrintDeviceTensor(variantPack.outTensors.at(i));
        }
        for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
            FreeTensor(variantPack.inTensors.at(i));
        }
        for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
            FreeTensor(variantPack.outTensors.at(i));
        }

        aclrtFree(workSpace);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        state.ResumeTiming();
    }
}

BENCHMARK(Layer_Bloom_7B)->Iterations(100);
