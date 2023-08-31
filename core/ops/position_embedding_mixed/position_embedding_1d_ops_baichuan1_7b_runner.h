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
#ifndef POSITIONEMBEDDING_1D_OPS_BAICHUAN1_7B_RUNNER_H
#define POSITIONEMBEDDING_1D_OPS_BAICHUAN1_7B_RUNNER_H
#include "acltransformer/base/ops_runner.h"
#include "acltransformer/params/position_embedding.h"

namespace AclTransformer {
class PositionEmbedding1dOpsBaichuan17bRunner : public OpsRunner {
public:
    PositionEmbedding1dOpsBaichuan17bRunner(const PositionEmbeddingParam &param);
    virtual ~PositionEmbedding1dOpsBaichuan17bRunner();

private:
    PositionEmbeddingParam param_;
};
} // namespace AclTransformer
#endif