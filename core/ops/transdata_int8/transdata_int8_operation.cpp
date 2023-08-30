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
#include "acltransformer/ops/transdata_int8_operation.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/config.h"
#include "transdata_int8_runner_builder.h"

namespace AclTransformer {

const int TRANSDATA_INT8_INTENSOR_COUNT = 1;
const int TRANSDATA_INT8_OUTTENSOR_COUNT = 1;
static const size_t IDX_0 = 0;
static const size_t IDX_1 = 1;
static const size_t IDX_2 = 2;
static const size_t IDX_3 = 3;
static const int64_t DEFAULT_INT8_ALIGN = 32;

static int64_t RoundUp(const int64_t val, const int64_t align = DEFAULT_INT8_ALIGN)
{
    if (align == 0) {
        return -1;
    }
    return (val + align - 1) / align * align;
}

TransDataInt8Operation::TransDataInt8Operation(const TransDataInt8Param &param)
    : Operation("TransDataInt8Operation"), param_(param)
{
    runnerBuilders_ = {new TransDataInt8OpsRunnerBuilder(param_)};
}

TransDataInt8Operation::~TransDataInt8Operation() {}

uint64_t TransDataInt8Operation::GetInTensorCount() const { return TRANSDATA_INT8_INTENSOR_COUNT; }

uint64_t TransDataInt8Operation::GetOutTensorCount() const { return TRANSDATA_INT8_OUTTENSOR_COUNT; }

AsdOps::Status TransDataInt8Operation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                      AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    // outTensorDescs.at(0) = inTensors.at(0).desc;
    auto inDims = inTensors.at(0).desc.dims;
    AsdOps::SVector<int64_t> auxDims{0, 0, 0, 0};
    auxDims[IDX_0] = 1;
    auxDims[IDX_1] = RoundUp(inDims[IDX_0]);
    auxDims[IDX_2] = RoundUp(inDims[IDX_1]) / DEFAULT_INT8_ALIGN;
    auxDims[IDX_3] = DEFAULT_INT8_ALIGN;

    // inference output dims: [N, H', W'/16, 16] -> [N, W'/16, H', 16]
    outTensorDescs.at(0).dims = AsdOps::SVector<int64_t>({0, 0, 0, 0});
    outTensorDescs.at(0).dims[IDX_0] = auxDims[IDX_0];
    outTensorDescs.at(0).dims[IDX_1] = auxDims[IDX_2];
    outTensorDescs.at(0).dims[IDX_2] = auxDims[IDX_1];
    outTensorDescs.at(0).dims[IDX_3] = auxDims[IDX_3];
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).format = AsdOps::TENSOR_FORMAT_FRACTAL_NZ;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *TransDataInt8Operation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer