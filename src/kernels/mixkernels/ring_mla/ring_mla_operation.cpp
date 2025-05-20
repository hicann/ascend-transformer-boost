/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "atbops/params/params.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"

static constexpr int32_t LONG_SEQ_LEN = 128;

namespace AtbOps {
using namespace Mki;
class RINGMLAOperation : public OperationBase {
public:
    explicit RINGMLAOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto inDtype = tensorQ.desc.dtype;
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RINGMLA),
            "OpParam is invalid", return nullptr);
        auto param = AnyCast<OpParam::RINGMLA>(launchParam.GetParam());
        switch (param.type) {
//            case OpParam::RINGMLA::SPLIT_CACHE:
//                return GetKernelByName("MLAKernel");
            case OpParam::RINGMLA::PREFILL_SPLIT_CACHE:
//                return inDtype == TENSOR_DTYPE_BF16 ? GetKernelByName("RINGMLAPrefillBF16Kernel") : GetKernelByName("RINGMLAPrefillKernel");
                return inDtype == TENSOR_DTYPE_BF16 ? GetKernelByName("RINGMLAPrefillKernel") : GetKernelByName("RINGMLAPrefillKernel");
            default:
                break;
        }
        MKI_LOG(ERROR) << "Unsupport RINGMLA type " << param.type;
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RINGMLA), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::RINGMLA>(specificParam);
        switch (param.type) {
//            case OpParam::RINGMLA::SPLIT_CACHE:
//                return DIM_8;
            case OpParam::RINGMLA::PREFILL_SPLIT_CACHE:
                return DIM_16;
            default:
                break;
        }
        return DIM_1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RINGMLA), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::RINGMLA>(specificParam);
        switch (param.type) {
            case OpParam::RINGMLA::SPLIT_CACHE:
                return DIM_2;
            case OpParam::RINGMLA::PREFILL_SPLIT_CACHE:
                return DIM_2;
            default:
                break;
        }
        return DIM_1;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RINGMLA),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::RINGMLA>(launchParam.GetParam());
        switch (param.type) {
//            case OpParam::RINGMLA::SPLIT_CACHE:
//                return InferShapeMLA(launchParam, outTensors);
            case OpParam::RINGMLA::PREFILL_SPLIT_CACHE:
                return InferShapeMLAPrefill(launchParam, outTensors);
            default:
                break;
        }
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Param is invalid");
    }

private:
    struct ShapeParam {
        int32_t maxQ;
        int32_t maxKv;
        uint32_t batch;
    };

//    Status InferShapeMLA(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
//    {
//        auto param = AnyCast<OpParam::RINGMLA>(launchParam.GetParam());
//        MKI_CHECK(CheckMLA(launchParam), "Failed to check launch param",
//            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
//        auto &tensorQ = launchParam.GetInTensor(DIM_0);
//        auto &tensorQRope = launchParam.GetInTensor(DIM_1);
//        outTensors[DIM_0].desc = tensorQ.desc;
//        outTensors[DIM_0].desc.dtype = tensorQRope.desc.dtype;
//        if (param.isRing) {
//            // outTensor1  lse
//            outTensors[DIM_1].desc = tensorQ.desc;
//            if (tensorQ.desc.dims.size() == DIM_2) {
//                outTensors[DIM_1].desc.dims[DIM_1] = DIM_1;
//            } else if (tensorQ.desc.dims.size() == DIM_3) {
//                outTensors[DIM_1].desc.dims[DIM_2] = DIM_1;
//            } else {
//                outTensors[DIM_1].desc.dims[DIM_0] = DIM_0;
//            }
//        } else {
//            outTensors[DIM_1].desc.dtype = tensorQ.desc.dtype;
//            outTensors[DIM_1].desc.format = tensorQ.desc.format;
//            outTensors[DIM_1].desc.dims = {0};
//        }
//        return Status::OkStatus();
//    }

    bool CheckPrefillBatchwise(const OpParam::RINGMLA &param, const Tensor &tensorQ) const
    {
        uint32_t batch = param.kvSeqLen.size();
        MKI_CHECK(param.kTensorList.size() == batch && param.vTensorList.size() == batch, "kv batch num invalid",
                  return false);
        auto kShape = param.kTensorList[0].desc.dims;
        MKI_CHECK(kShape.size() == 2, "kv batch shape size wrong", return false);
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

    bool CheckPrefilShape(const Mki::Tensor &tensorKcache, const Mki::Tensor &tensorVcache, const Mki::Tensor &tensorQ) const
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
                     const OpParam::RINGMLA &param) const
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
        bool alibi = param.maskType == OpParam::RINGMLA::MASK_TYPE_ALIBI;

        auto norm = param.maskType == OpParam::RINGMLA::MASK_TYPE_NORM;
        auto lookAhead = param.maskType == OpParam::RINGMLA::MASK_TYPE_LOOK_AHEAD;
        auto isLongSeq = (param.isTriuMask == 1) && (maskLen == LONG_SEQ_LEN);
        auto kvHead = param.kvHead == 0 ? headSize : param.kvHead;
        auto isAlibiCompress = maskLen == LONG_SEQ_LEN && currentShape[sz - DIM_2] != maskLen && alibi;
        // auto isSwaCompress = param.maskType == OpParam::RINGMLA::MASK_TYPE_SWA_COMPRESS;
        std::vector<std::pair<SVector<int64_t>, bool>> supports = {
            {{maxQ, maxKv}, true},
            {{LONG_SEQ_LEN, LONG_SEQ_LEN}, isLongSeq},
            {{batch, LONG_SEQ_LEN, LONG_SEQ_LEN}, isLongSeq},
            {{longSeqAlibiLen, longSeqAlibiLen}, alibi && sz == DIM_2},
            {{q.desc.dims[DIM_0], maxKv}, lookAhead},
            {{batch, maxQ, maxKv}, norm},
            {{headSize, maxQ, maxKv}, alibi},
            // {{static_cast<int32_t>(batch) / kvHead, maxQ, maxKv}, norm && param.compressHead},
            // {{headSize, maxQ, maxKv}, alibi && alibi && param.compressHead},
            {{headSize, maxQ, LONG_SEQ_LEN}, isAlibiCompress},
            {{batch, headSize, maxQ, maxKv}, true}
            // {{static_cast<int32_t>(batch) / kvHead, headSize, maxQ, maxKv}, alibi && param.compressHead},
            // {{SWA_COMPRESS_MASK_SIZE, SWA_COMPRESS_MASK_SIZE}, isSwaCompress}
        };
        // 保证mask一定能覆盖S，核内不会出现异常，用户保证1.避免多传;2.数值正常
        MKI_CHECK(FindMask(supports, currentShape, false), "current mask shape is unsupported!", return false);
        return true;
    }

    bool CheckMask(const LaunchParam &launchParam, Tensor q, Tensor mask) const
    {
        auto param = AnyCast<OpParam::RINGMLA>(launchParam.GetParam());
        auto head = param.headSize;
        auto qSeqLen = param.qSeqLen;
        auto kvSeqLen = param.kvSeqLen;
        uint32_t batch = kvSeqLen.size();
        auto maxQSeqlenIter = std::max_element(qSeqLen.begin(), qSeqLen.end());
        auto maxQ = maxQSeqlenIter != qSeqLen.end() ? *maxQSeqlenIter : 1;
        if (param.maskType == OpParam::RINGMLA::MASK_TYPE_NONE) {
            MKI_CHECK(CheckEmptyTensor(mask), "mask type inconsistent", return false);
        } else {
            MKI_CHECK(!CheckEmptyTensor(mask), "mask type inconsistent", return false);
        }
        if (CheckEmptyTensor(mask)) {
            return true;
        }
        // MKI_CHECK_NO_LOG(CheckMaskPre(mask, q), return false);
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

    Status InferShapeMLAPrefill(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        auto param = AnyCast<OpParam::RINGMLA>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0); // Q.shape = [batch*seqlen, hiddensize]
        auto &tensorlse = launchParam.GetInTensor(DIM_15);
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or bfloat16", return Status::FailStatus(ERROR_INVALID_VALUE));
        auto &tensorKcache = launchParam.GetInTensor(DIM_2); // K.shape = [layerNum, batch, max_seqlen, hiddensize]
        auto &tensorVcache = launchParam.GetInTensor(DIM_4); // V.shape = [layerNum, batch, max_seqlen, hiddensize]
        if (CheckEmptyTensor(tensorKcache) || CheckEmptyTensor(tensorVcache)) {
            MKI_CHECK(CheckEmptyTensor(tensorKcache) && CheckEmptyTensor(tensorVcache),
                      "normal k and v should both be empty tensor if batches are split", return Status::FailStatus(ERROR_INVALID_VALUE));
            MKI_CHECK(CheckPrefillBatchwise(param, tensorQ), "kv batchwise settings invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        } else {
            MKI_CHECK(CheckPrefilShape(tensorKcache, tensorVcache, tensorQ), "check datashape invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        }
        auto &tensorLayerId = launchParam.GetInTensor(DIM_5);
        MKI_CHECK(CheckEmptyTensor(tensorLayerId) || tensorLayerId.desc.dims.size() == 1,
                  "Input3 dim num should be 1 or null tensor", return Status::FailStatus(ERROR_INVALID_VALUE));
        auto ret = CheckMask(launchParam, tensorQ, launchParam.GetInTensor(DIM_6));
        MKI_CHECK(ret, "check mask shape fail", return Status::FailStatus(ERROR_INVALID_VALUE));
        if (tensorQ.desc.dims.size() == DIM_3) {
            auto embed = (tensorQ.desc.dims).at(2);
            outTensors[DIM_0].desc = tensorQ.desc;
            outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] = embed;

            outTensors[DIM_1].desc = tensorlse.desc;
        } else {
            auto embed = (launchParam.GetInTensor(0).desc.dims).at(1) / param.headSize;
            outTensors[DIM_0].desc = tensorQ.desc;
            outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] = embed * param.headSize;

            outTensors[DIM_1].desc = tensorlse.desc;
        }
        return Status::OkStatus();
    }


    bool CheckMLA(const LaunchParam &launchParam) const
    {
        return true;
    }
};

REG_OPERATION(RINGMLAOperation);
} // namespace AtbOps
