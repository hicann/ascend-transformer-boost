/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H
#define ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H

#include <mki/utils/log/log.h>

#define CHECK_DEVICE_VERSION_ASCEND910B()                                                                              \
    do {                                                                                                               \
        if (!(IsAscend910B())) {                                                                                       \
            MKI_LOG(WARN) << "check device, not in 910b, skip testcase";                                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_NOT_ASCEND910B()                                                                          \
    do {                                                                                                               \
        if ((IsAscend910B())) {                                                                                        \
            MKI_LOG(WARN) << "check device, in 910b, skip testcase";                                                   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_ASCEND910A()                                                                              \
    do {                                                                                                               \
        if (!(IsAscend910A())) {                                                                                       \
            MKI_LOG(WARN) << "check device, not in 910a, skip testcase";                                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_NOT_ASCEND910A()                                                                          \
    do {                                                                                                               \
        if ((IsAscend910A())) {                                                                                        \
            MKI_LOG(WARN) << "check device, in 910a, skip testcase";                                                   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_ASCEND310P()                                                                              \
    do {                                                                                                               \
        if (!(IsAscend310P())) {                                                                                       \
            MKI_LOG(WARN) << "check device, not in 310p, skip testcase";                                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_NOT_ASCEND310P()                                                                          \
    do {                                                                                                               \
        if ((IsAscend310P())) {                                                                                        \
            MKI_LOG(WARN) << "check device, in 310p, skip testcase";                                                   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_ASCEND310B()                                                                              \
    do {                                                                                                               \
        if (!(IsAscend310B())) {                                                                                       \
            MKI_LOG(WARN) << "check device, not in 310b, skip testcase";                                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_NOT_ASCEND310B()                                                                          \
    do {                                                                                                               \
        if ((IsAscend310B())) {                                                                                        \
            MKI_LOG(WARN) << "check device, in 310b, skip testcase";                                                   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

bool IsAscend310P();
bool IsAscend910A();
bool IsAscend910B();
bool IsAscend310B();
#endif // ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H