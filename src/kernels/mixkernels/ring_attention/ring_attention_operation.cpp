/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <numeric>
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"

static constexpr int32_t LONG_SEQ_LEN = 128;
static constexpr int32_t MLA_THRESHOLD = 256;
static constexpr int32_t SWA_COMPRESS_MASK_SIZE = 512;


namespace AtbOps {
using namespace Mki;
class RingAttentionOperation : public OperationBase {
public:
    explicit RingAttentionOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RingAttention), "OpParam is invalid",
                  return nullptr);
        auto param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        auto inDtype = launchParam.GetInTensor(0).desc.dtype;
        MKI_CHECK(param.headSize > 0, "headSize is invalid", return nullptr);
        auto embed = (launchParam.GetInTensor(0).desc.dims).at(1) / param.headSize; // headdim
        if (launchParam.GetInTensor(0).desc.dims.size() == DIM_3) {
            embed = (launchParam.GetInTensor(0).desc.dims).at(2); // 2 is head_size dim
        }
        auto isMla = (param.headDimV != 0 && (embed != param.headDimV || embed > MLA_THRESHOLD)) ? true : false;
        switch (param.type) {
            case OpParam::RingAttention::RING_ATTENTION:
                return GetKernelByName("RingAttentionKernel");
            default:
                break;
        }
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RingAttention), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::RingAttention>(specificParam);
        switch (param.type) {
            case OpParam::RingAttention::RING_ATTENTION:
                return DIM_14;
            default:
                break;
        }
        return DIM_14;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RingAttention), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::RingAttention>(specificParam);
        switch (param.type) {
            case OpParam::RingAttention::RING_ATTENTION:
                return DIM_2;
            default:
                break;
        }
        return DIM_2; // 1 output
    }

    Status InferShapeCheck(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RingAttention), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        MKI_LOG(INFO) << "infer shape param: " << param.headSize << ", type:" << param.type;
        auto scaleType = param.scaleType;
        MKI_CHECK(scaleType == OpParam::RingAttention::ScaleType::SCALE_LOGN_FP32 ||
                      scaleType == OpParam::RingAttention::ScaleType::SCALE_TOR,
                  "scaletype invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        if (scaleType == OpParam::RingAttention::ScaleType::SCALE_LOGN_FP32) {
            if (!CheckEmptyTensor(launchParam.GetInTensor(DIM_3))) {
                MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dims.at(0) != 0,
                          "logN is can not support prefix Cache", return Status::FailStatus(ERROR_INVALID_VALUE));
            }
            auto &tensorLog = launchParam.GetInTensor(DIM_11);
            MKI_CHECK(!CheckEmptyTensor(tensorLog), "Input12 should not be null tensor",
                      return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Input12 should not be null tensor"));
            auto inDtype = tensorLog.desc.dtype;
            MKI_CHECK(inDtype == TENSOR_DTYPE_FLOAT, "LogNscale should be float",
                      return Status::FailStatus(ERROR_INVALID_VALUE, "LogNscale should be float"));
            MKI_CHECK(param.dataShapeType == OpParam::RingAttention::DataShapeType::TYPE_BSND &&
                          !param.compressHead && param.windowSize == 0,
                      "LogN can not support BNSD,SWA,compressHead", return Status::FailStatus(ERROR_INVALID_VALUE));
        }
        MKI_CHECK(InferKVHead(launchParam), "kvHead is missing in OpParam for GQA",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "kvHead is missing in OpParam for GQA"));
        return Status::OkStatus();
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        if (!InferShapeCheck(launchParam).Ok()) {
            return Status::FailStatus(ERROR_INVALID_VALUE);
        }
        auto param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::RingAttention::RING_ATTENTION:
                return InferShapeRingAttention(launchParam, outTensors);
        }
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Param is invalid");
    }

private:
    struct ShapeParam {
        int32_t maxQ;
        int32_t maxKv;
        uint32_t batch;
    };

    Status InferOutTensorsRingAttention(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        auto param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(param.headSize > 0, "headSize is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        auto qSeqLen = param.qSeqLen;
        auto tokenNum = std::accumulate(qSeqLen.begin(), qSeqLen.end(), 0);
        if (param.dataShapeType == 1) {
            outTensors[DIM_0].desc = tensorQ.desc;
            if (tensorQ.desc.dims.size() == DIM_4) {
                // q 的shape b,s,n,d encoder
                auto embed = (tensorQ.desc.dims).at(3);
                outTensors[DIM_0].desc = tensorQ.desc;
                outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] =
                    param.headDimV == 0 ? embed : param.headDimV;

                outTensors[DIM_1].desc = tensorQ.desc;
                outTensors[DIM_1].desc.dims[DIM_1] = DIM_1;
            } else {
                // q 的shape bs,nd decoder
                auto embed = (launchParam.GetInTensor(0).desc.dims).at(1) / param.headSize;
                outTensors[DIM_0].desc = tensorQ.desc;
                outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] =
                param.headDimV == 0 ? embed * param.headSize : param.headDimV * param.headSize;

                outTensors[DIM_1].desc = tensorQ.desc;
                outTensors[DIM_1].desc.dims[DIM_1] = DIM_1;
            }
        } else {
            if (tensorQ.desc.dims.size() == DIM_3) {
                auto embed = (tensorQ.desc.dims).at(2);
                outTensors[DIM_0].desc = tensorQ.desc;
                outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] =
                param.headDimV == 0 ? embed : param.headDimV;

                outTensors[DIM_1].desc = tensorQ.desc;
                outTensors[DIM_1].desc.dims.resize(DIM_2);
                outTensors[DIM_1].desc.strides.resize(DIM_2);
                outTensors[DIM_1].desc.strides[DIM_0] = param.headSize;
                outTensors[DIM_1].desc.strides[DIM_1] = DIM_1;
                outTensors[DIM_1].desc.dims[DIM_0] = tensorQ.desc.dims[DIM_0];
                outTensors[DIM_1].desc.dims[DIM_1] = param.headSize;
            } else {
                auto embed = (launchParam.GetInTensor(0).desc.dims).at(1) / param.headSize;
                outTensors[DIM_0].desc = tensorQ.desc;
                outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] =
                param.headDimV == 0 ? embed * param.headSize : param.headDimV * param.headSize;

                outTensors[DIM_1].desc = tensorQ.desc;
                outTensors[DIM_1].desc.dims[DIM_1] = DIM_1;
            }
        }
        if (tensorQ.desc.dtype == TENSOR_DTYPE_INT8) {
            MKI_CHECK(param.outDataType == TENSOR_DTYPE_FLOAT16 || param.outDataType == TENSOR_DTYPE_BF16,
                      "param.outDataType is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
            outTensors[DIM_0].desc.dtype = param.outDataType;
        }
        MKI_LOG(INFO) << "InferOutTensorsRingAttention return True";
        return Status::OkStatus();
    }

    bool InferKVHead(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        // kvHead不传时默认为0，若不为0表示已经传入
        MKI_CHECK_NO_LOG(param.kvHead == 0, return true);
        // , head * embed) DIM_2
        if (launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_2) {
            auto headMulEmbed = (launchParam.GetInTensor(DIM_0).desc.dims).at(DIM_1);
            auto embed = headMulEmbed / param.headSize;
            auto kvHeadMulEmbedIdx = launchParam.GetInTensor(DIM_1).desc.dims.size() - 1;
            auto isMla = (param.headDimV != 0 && (embed != param.headDimV || embed > MLA_THRESHOLD)) ? true : false;
            if (launchParam.GetInTensor(DIM_1).desc.dims.size() > 0 && !isMla) {
                auto kvHeadMulEmbed = (launchParam.GetInTensor(DIM_1).desc.dims).at(kvHeadMulEmbedIdx);
                if (kvHeadMulEmbed > embed) {
                    // kvHead不传时默认为非GQA场景，否则返回kvHead缺失信息
                    MKI_CHECK_NO_LOG(headMulEmbed == kvHeadMulEmbed, return false);
                }
            }
        }
        // , head, embed) DIM_3 / BNSD DIM_4
        if (launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_3 ||
            launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_4) {
            auto realKVHeadIdx = launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_3 ?
                                      launchParam.GetInTensor(DIM_1).desc.dims.size() - 2 :
                                      launchParam.GetInTensor(DIM_1).desc.dims.size() - 3;
            auto embedIdx = launchParam.GetInTensor(DIM_0).desc.dims.size() - 1;
            auto embed = (launchParam.GetInTensor(DIM_0).desc.dims).at(embedIdx);
            auto isMla = (param.headDimV != 0 && (embed != param.headDimV || embed > MLA_THRESHOLD)) ? true : false;
            if (launchParam.GetInTensor(DIM_1).desc.dims.size() > 0 && !isMla) {
                auto realKVHead = (launchParam.GetInTensor(DIM_1).desc.dims).at(realKVHeadIdx);
                // kvHead不传时默认为非GQA场景，否则返回kvHead缺失信息
                MKI_CHECK_NO_LOG(param.headSize == realKVHead, return false);
            }
        }
        return true;
    }

    // InferShapeRingAttention
    Status InferShapeRingAttention(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckRingAttention(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsRingAttention(launchParam, outTensors));
        MKI_LOG(INFO) << "InferShapeRingAttention return True";
        return Status::OkStatus();
    }

    bool CheckQuant(const LaunchParam &launchParam) const
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1);
        auto &tensorVcache = launchParam.GetInTensor(DIM_2);
        auto param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        MKI_CHECK(param.dataShapeType == 0, "only support BSND", return false);
        MKI_CHECK(param.scaleType == 0, "not support LOGN", return false);
        const int32_t index1 = 1;
        const int32_t index2 = 2;
        auto embed = (launchParam.GetInTensor(0).desc.dims).at(index1) / param.headSize;
        if (launchParam.GetInTensor(0).desc.dims.size() == DIM_3) {
            embed = (launchParam.GetInTensor(0).desc.dims).at(index2);
        }
        auto isMla = (param.headDimV != 0 && (embed != param.headDimV || embed > MLA_THRESHOLD)) ? true : false;
        MKI_CHECK(!isMla, "mla not support in quant", return false);
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_INT8 && tensorKcache.desc.dtype == TENSOR_DTYPE_INT8 &&
                      tensorVcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Q, K, V should be int8", return false);

        auto &tensorQKscale = launchParam.GetInTensor(DIM_6);
        auto &tensorQKoffset = launchParam.GetInTensor(DIM_7);
        auto &tensorPVscale = launchParam.GetInTensor(DIM_8);
        auto &tensorPVoffset = launchParam.GetInTensor(DIM_9);

        MKI_CHECK(!CheckEmptyTensor(tensorQKscale) && !CheckEmptyTensor(tensorPVscale), "need SymmetricQuant scales",
                  return false);
        MKI_CHECK(CheckEmptyTensor(tensorQKoffset) && CheckEmptyTensor(tensorPVoffset), "only support SymmetricQuant",
                  return false);

        MKI_CHECK(tensorQKscale.desc.dtype == TENSOR_DTYPE_FLOAT && tensorPVscale.desc.dtype == TENSOR_DTYPE_FLOAT,
                  "QKscale, KVscale should be float", return false);

        MKI_CHECK(tensorQKscale.desc.dims[DIM_0] == param.headSize, "QKscale dim invalid, should be " << param.headSize,
                  return false);
        MKI_CHECK(tensorPVscale.desc.dims[DIM_0] == param.headSize,
                  "tensorPVscale dim invalid, should be " << param.headSize, return false);
        if (param.quantType == OpParam::RingAttention::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            auto &tensorOfflineQuant = launchParam.GetInTensor(DIM_10);
            MKI_CHECK(!CheckEmptyTensor(tensorOfflineQuant), "need offline quant scales", return false);
            MKI_CHECK(tensorOfflineQuant.desc.dtype == TENSOR_DTYPE_FLOAT, "offline scale should be float",
                      return false);
            MKI_CHECK(tensorOfflineQuant.desc.dims[DIM_0] == param.headSize,
                      "OfflineQuant scale dim invalid, should be " << param.headSize, return false);
        }
        return true;
    }

    bool CheckUnpadFAKvBatchwise(const OpParam::RingAttention &param, const Tensor &tensorQ) const
    {
        uint32_t batch = param.kvSeqLen.size();
        MKI_CHECK(param.kTensorList.size() == batch && param.vTensorList.size() == batch, "kv batch num invalid",
                  return false);
        auto kShape = param.kTensorList[0].desc.dims;
        if (param.dataShapeType == 1) {
            // todo N,S,D
            MKI_CHECK(kShape.size() == 3,
                      "kv batch shape size wrong,bnsd tensorList shape is [headNum,maxSeq,headDim] ", return false);
        } else {
            // max_kv_seqlen,hiddenSize
            MKI_CHECK(kShape.size() == 2, "kv batch shape size wrong", return false);
        }

        for (size_t i = 0; i < batch; ++i) {
            MKI_CHECK(param.kTensorList[i].data != nullptr && param.vTensorList[i].data != nullptr,
                      "kv batchwise ptr cannot be nullptr", return false);
            MKI_CHECK(param.kTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                          param.kTensorList[i].desc.dtype == TENSOR_DTYPE_BF16 ||
                          param.kTensorList[i].desc.dtype == TENSOR_DTYPE_INT8,
                      "k dtype invalid, should be either fp16 or bf16 or int8", return false);
            MKI_CHECK(param.vTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                          param.vTensorList[i].desc.dtype == TENSOR_DTYPE_BF16 ||
                          param.vTensorList[i].desc.dtype == TENSOR_DTYPE_INT8,
                      "v dtype invalid, should be either fp16 or bf16 or int8", return false);
            MKI_CHECK(param.kTensorList[i].desc.dtype == tensorQ.desc.dtype &&
                          param.vTensorList[i].desc.dtype == tensorQ.desc.dtype,
                      "Q K V dtype should be the same", return false);
        }
        return true;
    }

    bool CheckRingAttention(const LaunchParam &launchParam) const
    {
        // Q.shape = [num_tokens, num_heads, head_size] or [num_tokens. num_heads,* head_size]
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16 ||
                  tensorQ.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or BF16 or int8",
        return false);
        MKI_CHECK(tensorQ.desc.dims.size() == DIM_3 || tensorQ.desc.dims.size() == DIM_2 ||
                  tensorQ.desc.dims.size() == DIM_4,
                  "Input0 dim num " << tensorQ.desc.dims.size() << " invalid, dim should be 2 or 3 or 4", return false);

        auto &tensorK = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(tensorK.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorK.desc.dtype == TENSOR_DTYPE_BF16 ||
                  tensorK.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input1 dtype " << tensorK.desc.dtype << " invalid, should be float16 or BF16 or int8", return false);
        auto &tensorV = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(tensorV.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorV.desc.dtype == TENSOR_DTYPE_BF16 ||
                  tensorV.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input2 dtype " << tensorV.desc.dtype << " invalid, should be float16 or BF16 or int8", return false);
        MKI_CHECK(tensorK.desc.dims.size() == DIM_3 || tensorK.desc.dims.size() == DIM_2 ||
                  tensorK.desc.dims.size() == DIM_4,
                  "Input1 dim num " << tensorK.desc.dims.size() << " invalid, dim should be 2 or 3 or 4", return false);
        MKI_CHECK(tensorV.desc.dims.size() == DIM_3 || tensorV.desc.dims.size() == DIM_2 ||
                  tensorV.desc.dims.size() == DIM_4,
                  "Input2 dim num " << tensorV.desc.dims.size() << " invalid, dim should be 2 or 3 or 4", return false);
        MKI_CHECK((tensorV.desc.dims.size() == tensorK.desc.dims.size()) &&
                  (tensorQ.desc.dims.size() == tensorK.desc.dims.size()),
                  "The shape of input0 input1 and input2 should be same", return false);
        if (param.quantType == OpParam::RingAttention::QuantType::TYPE_QUANT_QKV_ONLINE ||
            param.quantType == OpParam::RingAttention::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            MKI_CHECK(CheckQuant(launchParam), "check quant type fail", return false);
        }
        auto &tensorMask = launchParam.GetInTensor(DIM_4);
        return CheckMask(launchParam, tensorQ, tensorMask);
    }

    bool CheckBSND(const Mki::Tensor &tensorKcache, const Mki::Tensor &tensorVcache, const Mki::Tensor &tensorQ) const
    {
        static const size_t KV_CACHE_DIM_NUM = 4;
        MKI_CHECK(tensorKcache.desc.dims.size() == KV_CACHE_DIM_NUM,
                  "Input1 dim num " << tensorKcache.desc.dims.size() << " invalid, should be " << KV_CACHE_DIM_NUM,
                  return false);
        MKI_CHECK(tensorVcache.desc.dims.size() == KV_CACHE_DIM_NUM,
                  "Input2 dim num " << tensorVcache.desc.dims.size() << " invalid, should be " << KV_CACHE_DIM_NUM,
                  return false);
        auto batch = tensorKcache.desc.dims[DIM_1];
        auto maxSeqlen = tensorKcache.desc.dims[DIM_2];
        MKI_CHECK(tensorVcache.desc.dims[DIM_1] == batch && tensorVcache.desc.dims[DIM_2] == maxSeqlen,
                  "Shape of input1 should be batch  input2 should be maxSeqlen", return false);
        MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorKcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorKcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype)
                                  << " invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorVcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorVcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype)
                                  << " invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorQ.desc.dtype == tensorKcache.desc.dtype && tensorKcache.desc.dtype == tensorVcache.desc.dtype,
                  "tensorQ K V must be the same dtype,here Q dtype is "
                      << GetStrWithDType(tensorQ.desc.dtype) << ", K dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype) << ", and V dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype),
                  return false);
        return true;
    }

    bool CheckBNSD(const Mki::Tensor &tensorKcache, const Mki::Tensor &tensorVcache, const Mki::Tensor &tensorQ) const
    {
        static const size_t BNSD_KV_CACHE_DIM_NUM = 5;
        MKI_CHECK(tensorKcache.desc.dims.size() == BNSD_KV_CACHE_DIM_NUM,
                  "BNSD:Input1 dim num " << tensorKcache.desc.dims.size() << " invalid, should be "
                                         << BNSD_KV_CACHE_DIM_NUM,
                  return false);
        MKI_CHECK(tensorVcache.desc.dims.size() == BNSD_KV_CACHE_DIM_NUM,
                  "BNSD:Input2 dim num " << tensorVcache.desc.dims.size() << " invalid, should be "
                                         << BNSD_KV_CACHE_DIM_NUM,
                  return false);
        auto maxSeq = tensorKcache.desc.dims[DIM_3];
        auto batch = tensorKcache.desc.dims[DIM_1];
        auto headNum = tensorKcache.desc.dims[DIM_2];
        MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorKcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorKcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype)
                                  << " invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorVcache.desc.dims[DIM_1] == batch && tensorVcache.desc.dims[DIM_3] == maxSeq &&
                      tensorVcache.desc.dims[DIM_2] == headNum,
                  "Shape of input0/input1 should be same", return false);
        MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorVcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorVcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype)
                                  << "data type invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorQ.desc.dtype == tensorKcache.desc.dtype && tensorKcache.desc.dtype == tensorVcache.desc.dtype,
                  "tensorQ K V must be the same dtype,here Q Tensor dtype is "
                      << GetStrWithDType(tensorQ.desc.dtype) << ", K dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype) << ", and V dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype),
                  return false);
        return true;
    }

    bool FindMask(std::vector<std::pair<SVector<int64_t>, bool>> &pairs, SVector<int64_t> &curShape, bool nz) const
    {
        auto target = std::find_if(pairs.begin(), pairs.end(), [curShape, nz](std::pair<SVector<int64_t>, bool> iter) {
            if (!iter.second || curShape.size() != iter.first.size()) {
                return false;
            }
            uint32_t count = 0;
            for (int32_t i = curShape.size() - 1; i >= 0; i--, count++) {
                // batch, head应该完全一致，maxQ和maxKv保证能够覆盖
                if (count < (nz ? DIM_3 : DIM_2)) {
                    if (iter.first[i] > curShape[i]) {
                        return false;
                    }
                } else {
                    if (iter.first[i] != curShape[i]) {
                        return false;
                    }
                }
            }
            return true;
        });
        return target != pairs.end() && (!nz || (*target).first[DIM_3] == FP16_ALIGN_NUM);
    }

    bool CheckNdMask(const Tensor &tensorMask, Tensor &q, const ShapeParam &shapePara,
                     const OpParam::RingAttention &param) const
    {
        auto maxQ = shapePara.maxQ;
        auto maxKv = shapePara.maxKv;
        auto batch = shapePara.batch;
        auto headSize = param.headSize;
        constexpr int32_t longSeqAlibiLen = 256;
        auto currentShape = tensorMask.desc.dims;
        auto sz = currentShape.size();
        MKI_CHECK(sz >= DIM_2, "mask invalid, please check.", return false);
        auto maskLen = currentShape[sz - 1];
        bool alibi = param.maskType == OpParam::RingAttention::MASK_TYPE_ALIBI;

        auto norm = param.maskType == OpParam::RingAttention::MASK_TYPE_NORM;
        auto lookAhead = param.maskType == OpParam::RingAttention::MASK_TYPE_LOOK_AHEAD;
        auto isLongSeq = (param.isTriuMask == 1) && (maskLen == LONG_SEQ_LEN);
        auto kvHead = param.kvHead == 0 ? headSize : param.kvHead;
        auto isAlibiCompress = maskLen == LONG_SEQ_LEN && currentShape[sz - DIM_2] != maskLen && alibi;
        auto isSwaCompress = param.maskType == OpParam::RingAttention::MASK_TYPE_SWA_COMPRESS;
        std::vector<std::pair<SVector<int64_t>, bool>> supports = {
            {{maxQ, maxKv}, true},
            {{LONG_SEQ_LEN, LONG_SEQ_LEN}, isLongSeq},
            {{batch, LONG_SEQ_LEN, LONG_SEQ_LEN}, isLongSeq},
            {{longSeqAlibiLen, longSeqAlibiLen}, alibi && sz == DIM_2},
            {{q.desc.dims[DIM_0], maxKv}, lookAhead},
            {{batch, maxQ, maxKv}, norm},
            {{headSize, maxQ, maxKv}, alibi},
            {{static_cast<int32_t>(batch) / kvHead, maxQ, maxKv}, norm && param.compressHead},
            {{headSize, maxQ, maxKv}, alibi && alibi && param.compressHead},
            {{headSize, maxQ, LONG_SEQ_LEN}, isAlibiCompress},
            {{batch, headSize, maxQ, maxKv}, true},
            {{static_cast<int32_t>(batch) / kvHead, headSize, maxQ, maxKv}, alibi && param.compressHead},
            {{SWA_COMPRESS_MASK_SIZE, SWA_COMPRESS_MASK_SIZE}, isSwaCompress}
        };
//         保证mask一定能覆盖S，核内不会出现异常，用户保证1.避免多传;2.数值正常
        MKI_CHECK(FindMask(supports, currentShape, false), "current mask shape is unsupported!", return false);
        return true;
    }

    bool CheckMaskPre(const Tensor &tensorMask, const Tensor &q) const
    {
        MKI_CHECK(q.desc.dtype == tensorMask.desc.dtype || q.desc.dtype == TENSOR_DTYPE_INT8,
                  "mask data type not consitent with q", return false);
        MKI_CHECK(tensorMask.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorMask.desc.dtype == TENSOR_DTYPE_BF16,
                  "mask dtype should be float16 or bfloat16", return false);
        return true;
    }

    bool CheckMask(const LaunchParam &launchParam, Tensor q, Tensor mask) const
    {
        auto param = AnyCast<OpParam::RingAttention>(launchParam.GetParam());
        auto head = param.headSize;
        auto qSeqLen = param.qSeqLen;
        auto kvSeqLen = param.kvSeqLen;
        uint32_t batch = kvSeqLen.size();
        auto maxQSeqlenIter = std::max_element(qSeqLen.begin(), qSeqLen.end());
        auto maxQ = maxQSeqlenIter != qSeqLen.end() ? *maxQSeqlenIter : 1;
        if (param.maskType == OpParam::RingAttention::MASK_TYPE_SWA_NORM && maxQ == 1) {
            MKI_CHECK(CheckEmptyTensor(mask), "mask type inconsistent", return false);
            // swa decode mask should be empty
        } else {
            if (param.maskType == OpParam::RingAttention::MASK_TYPE_NONE) {
                MKI_CHECK(CheckEmptyTensor(mask), "mask type inconsistent", return false);
            } else {
                MKI_CHECK(!CheckEmptyTensor(mask), "mask type inconsistent", return false);
            }
        }
        if (CheckEmptyTensor(mask)) {
            return true;
        }
        MKI_CHECK_NO_LOG(CheckMaskPre(mask, q), return false);
        MKI_CHECK(batch > 0, "batch invalid, please check", return false);
        // head
        MKI_CHECK(head > 0, "head invalid, please check", return false);
        // maxKvSeqlen
        auto maxKvSeqlenIter = std::max_element(kvSeqLen.begin(), kvSeqLen.end());
        MKI_CHECK(maxKvSeqlenIter != kvSeqLen.end() && *maxKvSeqlenIter > 0, "kvSeqlen invalid, please check",
                  return false);
        auto minKvSeqlenIter = std::min_element(kvSeqLen.begin(), kvSeqLen.end());
        MKI_CHECK((minKvSeqlenIter != kvSeqLen.end() && *minKvSeqlenIter >= 0),
                  "kvSeqlen min value invalid, please check", return false);
        // maxQSeqlen
        MKI_CHECK(maxQ > 0, "qSeqlen max value invalid, please check", return false);
        auto minQSeqlenIter = std::min_element(qSeqLen.begin(), qSeqLen.end());
        MKI_CHECK((minQSeqlenIter == qSeqLen.end()) || ((minQSeqlenIter != qSeqLen.end() && *minQSeqlenIter >= 0)),
                  "qSeqlen min value invalid, please check", return false);
        MKI_LOG(INFO) << "[batch, head, maxQ, maxKv]: [" << batch << ", " << head << ", " << maxQ << ", "
                      << *maxKvSeqlenIter << "]";
        MKI_CHECK(*maxKvSeqlenIter >= maxQ, "maxQ & maxKv inconsistent.", return false);
        ShapeParam shapePara = {maxQ, *maxKvSeqlenIter, batch};
        return CheckNdMask(mask, q, shapePara, param);
    }
};

REG_OPERATION(RingAttentionOperation);
} //    namespace AtbOps