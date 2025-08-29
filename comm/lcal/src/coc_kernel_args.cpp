/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coc_kernel_args.h"
#include <sstream>
#include <string>
#include <mki/utils/log/log.h>
#include <mki/utils/rt/other/other.h>
#include "tiling.h"
#include "lcal_internal.h"
using namespace Mki;

namespace Lcal {
int CoCkernelArgs::SetFFTSAddr()
{
    uint32_t fftsLen;
    int MkiRtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    if (error != MKIRT_SUCCESS) {
        MKT_LOG(ERROR) << "MkiRtGetC2cCtrlAddr err";
        return LCAL_ERROR_MKIRT;
    }
    return LCAL_SUCCESS;    
}

void CoCkernelArgs::SetInputPkgArgs(CoCInputPkg &inputPkg)
{
    matrixA = inputPkg.matrixA;
    matrixB = inputPkg.matrixB;
    bias = inputPkg.bias;
    gamma= inputPkg.gamma;
    dequantScale = inputPkg.dequantScale;
    dequantOffset = inputPkg.dequantOffset;
    quantScale = inputPkg.quantScale;
    quantOffset = inputPkg.quantOffset;
}

void CoCkernelArgs::SetOutputPkgArgs(CoCOutputPkg &outputPkg)
{
    output = outputPkg.output;
    minOutput = outputPkg.midOutput;
}

void CoCkernelArgs::SetWorkspacePtrArg(void *workspacePtr)
{
    workspace = workspacePtr;
}

void CoCkernelArgs::SetParamDescArgs(const CoCParamDesc &paramDesc)
{
    cockernelParm.quantInfo = paramDesc.quantInfo;
    cockernelParm.twoDimTPInfo = paramDesc.twoDimTPInfo;
    cockernelParm.postInfo = paramDesc.postInfo;
    cockernelParm.weightNz = paramDesc.mmInfo.weightNz;
}

void CoCkernelArgs::SetCommArgs(const LcalComm &comm)
{
    commArgsPtr = comm.GetCommArgsPtr();
}

void CoCkernelArgs::SetCoCTilingDataArgs(const CoCTilingData &tilingData)
{
    pCocTiling = &(cockernelParm.cocTilingData);
    cockernelParm.cocTilingData = tilingData;
}

std::string CoCkernelArgs::ParamToString()
{
    std::string quantInfoString = "[QuantInfo]: dequantGranularity=" + 
                                    std::to_string(cockernelParm.quantInfo.dequantGranularity) + "\n";
    std::string weightNzInfoString = "[weightNz]: weightNz=" + 
                                    std::to_string(cockernelParm.weightNz) + "\n";
    std::string tilingInfoString = cockernelParm.cocTilingData.ToString();
    return quantInfoString + weightNzInfoString + tilingInfoString;
}
}