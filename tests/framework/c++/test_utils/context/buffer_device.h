/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef BUFFER_DEVICE_H
#define BUFFER_DEVICE_H
#include "buffer_base.h"

namespace atb {
class BufferDevice : public BufferBase {
public:
    explicit BufferDevice(uint64_t bufferSize);
    ~BufferDevice() override;
    void *GetBuffer(uint64_t bufferSize) override;

private:
    void Free();

private:
    void *buffer_ = nullptr;
    uint64_t bufferSize_ = 0;
};
} // namespace atb
#endif