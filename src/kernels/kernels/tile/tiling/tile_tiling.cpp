/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "tile_tiling.h"
#include "asdops/params/expand.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status TileTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                    const BinHandle &binHandle)
{
    const TensorDesc &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;
    auto param = AnyCast<OpParam::Expand>(launchParam.GetParam());
    SVector<int64_t> shape = param.shape;
    SVector<int64_t> shapeDim = {static_cast<int64_t>(shape.size())};
    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Tile")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND,
                       shapeDim, shape.data(), shape.size() * sizeof(int64_t))
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

} // namespace AsdOps
