/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_tiling.h"
#include <dlfcn.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling_data.h"

namespace AtbOps {
const uint64_t BLOCK_DIM = 40;
Status FusionTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    OpParam::Fusion fusionType = launchParam.GetParam<OpParam::Fusion>();
    std::string path(std::getenv("HOME"));
    path += std::string("/.atb_auto_fusion/bishengir_bin/") +
            (OpParam::Fusion::MATMUL_ADD == fusionType.fusionType ? "libmatmul_add.so" : "libmatmul_gelu.so");
    std::string inferWorkspaceFuncName =
        +(OpParam::Fusion::MATMUL_ADD == fusionType.fusionType ? "matmul_add_" : "matmul_gelu_");
    FusionTilingData *tilingDataPtr = reinterpret_cast<FusionTilingData *>(kernelInfo.GetTilingHostAddr());
    void *handle = dlopen(path.c_str(), RTLD_LAZY);
    if (!handle) {
        MKI_LOG(ERROR) << "host tiling load error!";
        return Status::FailStatus(-1, "Can not open the binary");
    }
    TILING_FUNC_GET tilingFunc = nullptr;
    std::string tilingFuncName = inferWorkspaceFuncName + "tiling_func";
    *(void **)(&tilingFunc) = dlsym(handle, tilingFuncName.c_str());
    KernelArgs *kernelArgs = new (std::nothrow) KernelArgs;
    kernelArgs->tilingDevice = static_cast<void *>(tilingDataPtr);
    kernelArgs->tilingDeviceDup = kernelArgs->tilingDevice;
    tilingFunc(static_cast<void *>(kernelArgs));
    INFER_WORKSPACE_T inferworkspaceFunc;
    inferWorkspaceFuncName += std::to_string(tilingDataPtr->key) + "_infer_workspace_shape_function";
    MKI_LOG(INFO) << "now inferWorkspaceFuncName is" << inferWorkspaceFuncName;
    *(void **)(&inferworkspaceFunc) = dlsym(handle, inferWorkspaceFuncName.c_str());
    KernelArgsForInferShapeWorkspaceWithTiling *wsWithTiling =
        new (std::nothrow) KernelArgsForInferShapeWorkspaceWithTiling;
    wsWithTiling->tilingDevice = tilingDataPtr;
    wsWithTiling->tilingDeviceDup = tilingDataPtr;
    int64_t workSpaceSize = inferworkspaceFunc(static_cast<void *>(wsWithTiling));
    workSpaceSize *= sizeof(long long);
    MKI_LOG(INFO) << "please check workSpaceSize = " << workSpaceSize;
    kernelInfo.GetScratchSizes().push_back(static_cast<uint64_t>(workSpaceSize));
    kernelInfo.SetTilingId(tilingDataPtr->key);
    kernelInfo.SetBlockDim(BLOCK_DIM);
    delete kernelArgs;
    delete wsWithTiling;
    return Status::OkStatus();
}
} // namespace AtbOps
