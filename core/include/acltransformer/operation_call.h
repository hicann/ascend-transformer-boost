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
#ifndef ACLTRANSFORMER_OPERATIONCALL_H
#define ACLTRANSFORMER_OPERATIONCALL_H
#include <string>
#include <asdops/utils/any/any.h>
#include <asdops/utils/svector/svector.h>
#include <asdops/tensor.h>

namespace AclTransformer {
class Operation;
class Plan;

class OperationCall {
public:
    OperationCall(const std::string &opName, const AsdOps::Any &opParam);
    ~OperationCall();
    int ExecuteSync(const AsdOps::SVector<AsdOps::Tensor> &inTensors, const AsdOps::SVector<AsdOps::Tensor> &outTensors,
                    void *stream);
    int ExecuteAsync(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                     const AsdOps::SVector<AsdOps::Tensor> &outTensors, void *stream);

private:
    std::shared_ptr<Operation> operation_;
    std::shared_ptr<Plan> plan_;
};
} // namespace AclTransformer
#endif