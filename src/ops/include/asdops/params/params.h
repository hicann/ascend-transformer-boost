/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CORE_PARAMS_PARAMS_H
#define CORE_PARAMS_PARAMS_H
#include "asdops/params/common.h"
#include "asdops/params/asstrided.h"
#include "asdops/params/concat.h"
#include "asdops/params/copy.h"
#include "asdops/params/cumsum.h"
#include "asdops/params/expand.h"
#include "asdops/params/fill.h"
#include "asdops/params/gather.h"
#include "asdops/params/matmul.h"
#include "asdops/params/onehot.h"
#include "asdops/params/reduce.h"
#include "asdops/params/reverse.h"
#include "asdops/params/slice.h"
#include "asdops/params/sort.h"
#include "asdops/params/split.h"
#include "asdops/params/softmax.h"
#include "asdops/params/transdata.h"
#include "asdops/params/transpose.h"
#include "asdops/params/zeroslike.h"

#include "asdops/params/fastsoftmax.h"
#include "asdops/params/fastsoftmax_grad.h"
#include "asdops/params/ffn.h"
#include "asdops/params/genattentionmask.h"
#include "asdops/params/laser_attention.h"
#include "asdops/params/laser_attention_grad.h"
#include "asdops/params/pad_with_hidden_state.h"
#include "asdops/params/stridedbatchmatmul.h"
#include "asdops/params/moe_gmm.h"
#include "asdops/params/unpad_with_hidden_state.h"
#include "asdops/params/rope_grad.h"

#endif

// #include "asdops/params/activation.h"
// #include "asdops/params/elewise.h"
// #include "asdops/params/index.h"
// #include "asdops/params/multinomial.h"
// #include "asdops/params/norm.h"
// #include "asdops/params/dynamic_ntk.h"
// #include "asdops/params/group_topk.h"
