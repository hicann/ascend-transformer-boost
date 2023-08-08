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
#ifndef ACLTRANSFOERM_PARAMS_PARAMS_H
#define ACLTRANSFOERM_PARAMS_PARAMS_H
#include "acltransformer/params/add.h"
#include "acltransformer/params/add_norm.h"
#include "acltransformer/params/add_norm_quant.h"
#include "acltransformer/params/all_reduce.h"
#include "acltransformer/params/embedding.h"
#include "acltransformer/params/ffn.h"
#include "acltransformer/params/ffn_quant.h"
#include "acltransformer/params/linear.h"
#include "acltransformer/params/linear_parallel.h"
#include "acltransformer/params/linear_quant.h"
#include "acltransformer/params/lm_head.h"
#include "acltransformer/params/matmul.h"
#include "acltransformer/params/mlp.h"
#include "acltransformer/params/norm.h"
#include "acltransformer/params/norm_quant.h"
#include "acltransformer/params/position_embedding.h"
#include "acltransformer/params/position_embedding_1d_split.h"
#include "acltransformer/params/position_embedding_fusion.h"
#include "acltransformer/params/post.h"
#include "acltransformer/params/quant.h"
#include "acltransformer/params/rms_norm.h"
#include "acltransformer/params/self_attention.h"
#include "acltransformer/params/self_attention_kv_cache.h"
#include "acltransformer/params/self_attention_kv_cache_fusion.h"
#include "acltransformer/params/transpose.h"
#endif