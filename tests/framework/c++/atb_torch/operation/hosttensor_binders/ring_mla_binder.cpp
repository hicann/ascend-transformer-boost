/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ring_mla_binder.h"

#include <sstream>
#include "atb/utils/log.h"

namespace {
// 6: seqlen; ring MLA: query_split1, query_split2, key_split1, key_split2, value, mask, seqlen
const int SEQLEN_INDEX = 6;
} // namespace


RingMLABinder::RingMLABinder() {}

RingMLABinder::~RingMLABinder() {}

void RingMLABinder::ParseParam(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << "Ring MLA ParseParam";
    seqLen_.clear();
    for (const auto &item : paramJson["seqLen"]) {
        seqLen_.emplace_back(item.get<int32_t>());
    }
    ATB_LOG(INFO) << "Ring MLA seqLen size: " << seqLen_.size();
}


void RingMLABinder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(INFO) << "Ring MLA BindTensor";
    variantPack.inTensors.at(SEQLEN_INDEX).hostData = seqLen_.data();
}
 +26 tests/framework/c++/atb_torch/operation/hosttensor_binders/ring_mla_binder.h  0 -> 100644
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RING_MLA_BINDER_H
#define ATB_RING_MLA_BINDER_H
#include "hosttensor_binder.h"
#include <vector>

class RingMLABinder : public HostTensorBinder {
public:
    RingMLABinder();
    virtual ~RingMLABinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> seqLen_;
};

#endif // ATB_RING_MLA_BINDER_H
