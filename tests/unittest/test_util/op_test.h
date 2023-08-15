/**
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
#ifndef ACLTRANSFORMER_OPTEST_H
#define ACLTRANSFORMER_OPTEST_H
#include <functional>
#include <string>
#include <asdops/tensor.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/plan.h"
#include "acltransformer/operation.h"

namespace AclTransformer {
struct GoldenContext {
    AsdOps::SVector<AsdOps::Tensor> hostInTensors;
    AsdOps::SVector<AsdOps::Tensor> hostOutTensors;
    AsdOps::SVector<AsdOps::Tensor> deviceInTensors;
    AsdOps::SVector<AsdOps::Tensor> deviceOutTensors;
};

using OpTestGolden = std::function<AsdOps::Status(const GoldenContext &context)>;

class OpTest {
public:
    explicit OpTest();
    ~OpTest();
    void Golden(OpTestGolden golden);
    void FloatRand(float min, float max);
    void Int8Rand(int8_t min, int8_t max);
    void IntRand(int32_t min, int32_t max);
    void LongRand(int64_t min, int64_t max);
    AsdOps::Status Run(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs);
    AsdOps::Status Run(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::Tensor> &inTensorLists);
    AsdOps::Status Run(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::Tensor> &inTensorLists,
                       const AsdOps::Any &varaintPackParam);
    AsdOps::Status Run(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs,
                       const AsdOps::Any &varaintPackParam);
    AsdOps::Status RunImpl(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::Tensor> &inTensorLists,
                           const AsdOps::Any &varaintPackParam);
    AsdOps::Status RunImpl(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::Tensor> &inTensorLists);

private:
    void GenerateRandomTensors(const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs,
                               AsdOps::SVector<AsdOps::Tensor> &inTensors);
    void Init();
    void Cleanup();
    AsdOps::Status Prepare(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::Tensor> &inTensorLists);
    AsdOps::Tensor CreateHostTensor(const AsdOps::Tensor &tensorIn);
    AsdOps::Status CopyDeviceTensorToHostTensor();
    AsdOps::Tensor HostTensor2DeviceTensor(const AsdOps::Tensor &hostTensor);
    std::string TensorToString(const AsdOps::Tensor &tensor);
    AsdOps::Tensor CreateHostZeroTensor(const AsdOps::TensorDesc &tensorDesc);
    void BuildVariantPack(const AsdOps::SVector<AsdOps::Tensor> &inTensorLists);
    AsdOps::Status RunOperation();
    AsdOps::Status RunGolden();
    AsdOps::Tensor CreateHostRandTensor(const AsdOps::TensorDesc &tensorDesc);

private:
    int deviceId_ = 0;
    OpTestGolden golden_;
    GoldenContext goldenContext_;
    AclTransformer::Operation *operation_;
    AclTransformer::Plan plan_;
    AclTransformer::Handle handle_;
    AclTransformer::VariantPack variantPack_;
    float randFloatMin_ = 1;
    float randFloatMax_ = 1;
    int8_t randInt8Min_ = 1;
    int8_t randInt8Max_ = 1;
    int32_t randIntMin_ = 1;
    int32_t randIntMax_ = 1;
    int64_t randLongMin_ = 1;
    int64_t randLongMax_ = 1;
};
} // namespace AclTransformer
#endif