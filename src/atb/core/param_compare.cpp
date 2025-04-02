/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/core/param_compare.h"
#include <functional>
#include <map>
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include <mki/launch_param.h>
#include "atb/utils/log.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#include "atb/utils/tensor_util.h"

namespace atb {
using ParamCompareFunc = std::function<bool(const Mki::Any &, const Mki::Any &)>;

template <typename T> bool ParamCompareFuncImpl(const Mki::Any &any1, const Mki::Any &any2)
{
    const auto &content1 = Mki::AnyCast<T>(any1);
    const auto &content2 = Mki::AnyCast<T>(any2);
    return content1 == content2;
}

static const std::map<std::size_t, ParamCompareFunc> ASDOPS_PARAM_COMPARE_MAP = {
    {typeid(AtbOps::OpParam::Activation).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Activation>},
    {typeid(AsdOps::OpParam::AsStrided).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::AsStrided>},
    {typeid(AsdOps::OpParam::Concat).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Concat>},
    {typeid(AsdOps::OpParam::Copy).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Copy>},
    {typeid(AtbOps::OpParam::Elewise).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Elewise>},
    {typeid(AsdOps::OpParam::Expand).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Expand>},
    {typeid(AsdOps::OpParam::Fill).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Fill>},
    {typeid(AsdOps::OpParam::Gather).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Gather>},
    {typeid(AsdOps::OpParam::MatMul).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::MatMul>},
    {typeid(AtbOps::OpParam::Multinomial).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Multinomial>},
    {typeid(AtbOps::OpParam::Norm).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Norm>},
    {typeid(AsdOps::OpParam::Slice).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Slice>},
    {typeid(AsdOps::OpParam::Softmax).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Softmax>},
    {typeid(AsdOps::OpParam::Sort).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Sort>},
    {typeid(AsdOps::OpParam::Split).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Split>},
    {typeid(AsdOps::OpParam::Transdata).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Transdata>},
    {typeid(AsdOps::OpParam::Transpose).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Transpose>},
    {typeid(AsdOps::OpParam::Onehot).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Onehot>},
    {typeid(AsdOps::OpParam::Reduce).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Reduce>},
    {typeid(AsdOps::OpParam::FastSoftMaxGrad).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::FastSoftMaxGrad>},
    {typeid(AsdOps::OpParam::FastSoftMax).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::FastSoftMax>},
    {typeid(AtbOps::OpParam::Gating).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Gating>},
    {typeid(AsdOps::OpParam::GenAttentionMask).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::GenAttentionMask>},
    {typeid(AtbOps::OpParam::KVCache).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::KVCache>},
    {typeid(AtbOps::OpParam::Pad).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Pad>},
    {typeid(AsdOps::OpParam::PadWithHiddenState).hash_code(),
     ParamCompareFuncImpl<AsdOps::OpParam::PadWithHiddenState>},
    {typeid(AtbOps::OpParam::PagedAttention).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::PagedAttention>},
    {typeid(AtbOps::OpParam::ReshapeAndCache).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::ReshapeAndCache>},
    {typeid(AtbOps::OpParam::Rope).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Rope>},
    {typeid(AsdOps::OpParam::RopeGrad).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::RopeGrad>},
    {typeid(AsdOps::OpParam::StridedBatchMatmul).hash_code(),
     ParamCompareFuncImpl<AsdOps::OpParam::StridedBatchMatmul>},
    {typeid(AtbOps::OpParam::Toppsample).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Toppsample>},
    {typeid(AtbOps::OpParam::Unpad).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Unpad>},
    {typeid(AtbOps::OpParam::UnpadFlashAttentionNz).hash_code(),
     ParamCompareFuncImpl<AtbOps::OpParam::UnpadFlashAttentionNz>},
    {typeid(AtbOps::OpParam::UnpadFlashAttention).hash_code(),
     ParamCompareFuncImpl<AtbOps::OpParam::UnpadFlashAttention>},
    {typeid(AsdOps::OpParam::UnpadWithHiddenState).hash_code(),
     ParamCompareFuncImpl<AsdOps::OpParam::UnpadWithHiddenState>},
    {typeid(AsdOps::OpParam::Cumsum).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Cumsum>},
    {typeid(AtbOps::OpParam::DynamicNTK).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::DynamicNTK>},
    {typeid(AtbOps::OpParam::GroupTopk).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::GroupTopk>},
    {typeid(AtbOps::OpParam::Index).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::Index>},
    {typeid(AsdOps::OpParam::Reverse).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Reverse>},
    {typeid(AsdOps::OpParam::ZerosLike).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::ZerosLike>},
    {typeid(AtbOps::OpParam::BlockCopy).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::BlockCopy>},
    {typeid(AsdOps::OpParam::FFN).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::FFN>},
    {typeid(AtbOps::OpParam::GmmAdd).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::GmmAdd>},
    {typeid(AsdOps::OpParam::LaserAttention).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::LaserAttention>},
    {typeid(AsdOps::OpParam::LaserAttentionGrad).hash_code(),
     ParamCompareFuncImpl<AsdOps::OpParam::LaserAttentionGrad>},
    {typeid(AsdOps::OpParam::MoeGmm).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::MoeGmm>},
    {typeid(AtbOps::OpParam::FusedAddTopkDiv).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::FusedAddTopkDiv>},
    {typeid(AtbOps::OpParam::MlaPreprocess).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::MlaPreprocess>},
    {typeid(AtbOps::OpParam::SwigluQuant).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::SwigluQuant>},
    {typeid(AtbOps::OpParam::RopeQConcat).hash_code(), ParamCompareFuncImpl<AtbOps::OpParam::RopeQConcat>},
    {typeid(AtbOps::OpParam::RmsNormAndRopeAndReshapeAndCache).hash_code(),
     ParamCompareFuncImpl<AtbOps::OpParam::RmsNormAndRopeAndReshapeAndCache>},
};

bool IsLaunchParamEqual(const Mki::LaunchParam &launchParam1, const Mki::LaunchParam &launchParam2)
{
    if (launchParam1.GetInTensorCount() != launchParam2.GetInTensorCount()) {
        return false;
    }

    for (size_t i = 0; i < launchParam1.GetInTensorCount(); ++i) {
        if (!TensorUtil::AsdOpsTensorDescEqual(launchParam1.GetInTensor(i).desc, launchParam2.GetInTensor(i).desc)) {
            return false;
        }
    }

    const Mki::Any &specificParam1 = launchParam1.GetParam();
    const Mki::Any &specificParam2 = launchParam2.GetParam();
    auto it = ASDOPS_PARAM_COMPARE_MAP.find(specificParam1.Type().hash_code());
    if (it != ASDOPS_PARAM_COMPARE_MAP.end()) {
        return it->second(specificParam1, specificParam2);
    } else {
        ATB_LOG(WARN) << "Can not compare param of " << specificParam1.Type().name();
        return false;
    }
}
} // namespace atb