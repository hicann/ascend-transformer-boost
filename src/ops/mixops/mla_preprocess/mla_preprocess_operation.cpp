/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <algorithm>
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
 
namespace AtbOps {
const int32_t DIM_23 = 23;
const int32_t Q_HEADDIM = 576;
using namespace Mki;
class MlaPreprocessOperation : public OperationBase {
public:
    explicit MlaPreprocessOperation(const std::string &opName) noexcept : OperationBase(opName) {}
 
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MlaPreprocess), "Invalid Op Param.", return nullptr);
        auto inDtype = launchParam.GetInTensor(0).desc.dtype;
        if (inDtype == TENSOR_DTYPE_BF16) {
            return GetKernelByName("MLAPreprocessBF16Kernel");
        }
        return GetKernelByName("MLAPreprocessKernel");
    }
 
    int64_t GetInputNum(const Any &specificParam) const override
    {
        return DIM_23;
    }
    
    int64_t GetOutputNum(const Any &specificParam) const override
    {
        return DIM_2;
    }
 
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorKEY = launchParam.GetInTensor(16);
        auto param = AnyCast<OpParam::MlaPreprocess>(launchParam.GetParam());
        outTensors[DIM_0].desc = tensorQ.desc;
        outTensors[DIM_0].desc.dims[0] = param.N;
        outTensors[DIM_0].desc.dims[1] = param.headNum * Q_HEADDIM;
        outTensors[DIM_1].desc = tensorKEY.desc;
        return Mki::Status::OkStatus();
    }
};
 
REG_OPERATION(MlaPreprocessOperation);
} //    namespace AtbOps