/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H
#include <vector>
#include <string>
#include <atb/types.h>

atb::Tensor FillTensorDataRandomly(const atb::TensorDesc &desc, float range_min, float range_max);

atb::Tensor FillTensorDataRandomly(const atb::TensorDesc &desc);

atb::Tensor FillTensorDataRandomly(const atb::TensorDesc &desc, const std::pair<float, float> range);

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs);

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs,
                                                const std::vector<float> &range_mins,
                                                const std::vector<float> &range_maxs);
                                                
std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs, float range_min,
                                                float range_max);

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs,
                                                const std::vector<std::pair<float, float>> &ranges);

std::vector<atb::Tensor> FillTensorDataRandomly(const std::vector<atb::TensorDesc> &descs,
                                                std::pair<float, float> range);

atb::Tensor FillTensorDataByZero(const atb::TensorDesc &desc);

std::vector<atb::Tensor> FillTensorDataByZero(const std::vector<atb::TensorDesc> &descs);

atb::Tensor FillTensorDataByOne(const atb::TensorDesc &desc);

std::vector<atb::Tensor> FillTensorDataByOne(const std::vector<atb::TensorDesc> &descs);

atb::Tensor FillTensorDataByFile(const atb::TensorDesc &desc, const std::string &filePath);

std::vector<atb::Tensor> FillTensorDataByFile(const std::vector<atb::TensorDesc> &descs,
                                              const std::vector<std::string> &filePaths);

void FreeTensor(atb::Tensor &tensor);

void FreeTensor(std::vector<atb::Tensor> &tensors);

void PrintDeviceTensor(const atb::Tensor &tensor);

void PrintDeviceTensor(const std::vector<atb::Tensor> &tensors);

#endif