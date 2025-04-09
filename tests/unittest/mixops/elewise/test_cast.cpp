/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <mki/utils/log/log.h>
#include "atbops/params/elewise.h"
#include "device_version_check.h"
#include "test_utils/float_util.h"
#include "test_utils/atb_mixops_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AtbOps;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;

Status CastWideGolden2Int32(float atol, float rtol, const GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);

    at::Tensor atInRefTensor;
    at::Tensor atInRefTensorConverted;
    at::Tensor atOutRefTensor;
    at::Tensor atOutRefTensorConverted;

    if (inTensor.desc.dtype == TENSOR_DTYPE_FLOAT && outTensor.desc.dtype == TENSOR_DTYPE_INT32) {
        atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kFloat);
        atInRefTensorConverted = atInRefTensor.to(at::kInt);

        atOutRefTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kInt);
    } else if (inTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 && outTensor.desc.dtype == TENSOR_DTYPE_INT32) {
        atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
        atInRefTensorConverted = atInRefTensor.to(at::kInt);

        atOutRefTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kInt);
    } else {
        MKI_LOG(ERROR) << "Unsupported type conversion!!!" << GetStrWithDType(inTensor.desc.dtype)
            << " -> " << GetStrWithDType(outTensor.desc.dtype);
    }

    int32_t *atOutRefArray = (int32_t *)atInRefTensorConverted.storage().data_ptr().get();
    int32_t *atOutArray = (int32_t *)atOutRefTensor.storage().data_ptr().get();

    for (int64_t i = 0; i < inTensor.Numel(); i++) {
        int32_t expect = atOutRefArray[i];
        int32_t result = atOutArray[i];
        if (!FloatUtil::FloatJudgeEqual(expect, result, atol, rtol)) {
            std::string msg = "pos " + std::to_string(i) + ", expect: " + std::to_string(expect) +
                              ", result:" + std::to_string(result);
            return Status::FailStatus(-1, msg);
        }
    }
    return Status::OkStatus();
}

Status CastWideGoldenINT(float atol, float rtol, const GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    for (int64_t i = 0; i < inTensor.Numel(); i++) {
        int64_t expect = inTensor.desc.dtype == TENSOR_DTYPE_INT32 ? static_cast<int32_t *>(inTensor.data)[i]
                                                                   : static_cast<int64_t *>(inTensor.data)[i];
        int64_t result = outTensor.desc.dtype == TENSOR_DTYPE_INT32 ? static_cast<int32_t *>(outTensor.data)[i]
                                                                    : static_cast<int64_t *>(outTensor.data)[i];
        if (!FloatUtil::FloatJudgeEqual(expect, result, atol, rtol)) {
            std::string msg = "pos " + std::to_string(i) + ", expect: " + std::to_string(expect) +
                              ", result:" + std::to_string(result);
            return Status::FailStatus(-1, msg);
        }
    }
    return Status::OkStatus();
}

TEST(TestOpElewiseCast, Cast)
{
    MkiOpTest opTest;
    opTest.Golden(std::bind(&Golden::InOutTensorEqual, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    UtOpDesc opDesc = {"ElewiseOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10000000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10000000}});
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, InferShape0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5000}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5000}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestOpElewiseCast, InferShape1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5000}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5000}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_UNDEFINED;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestOpElewiseCast, CanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF16toF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpElewiseCast, CanSupport1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF32toF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpElewiseCast, CanSupport2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF32toF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCast, CanSupport3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF32toF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCast, CanSupport4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF32toF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCast, CanSupport5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF32toF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCast, CanSupport6)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF16toF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCast, CanSupport7)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_CAST;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF16toF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCast, CanSupport8)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastF16toF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCast, CanSupport9)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CastWideKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpElewiseCast, CastWideCase0) // FLOAT16 -> FLOAT32 【cast_half_to_float;   //0001 0001   17】
{
    MkiOpTest opTest;
    opTest.Golden(std::bind(&Golden::InOutTensorEqual, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_FLOAT;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase1) // FLOAT32 -> FLOAT16 【cast_float_to_half;    // 0010 0001   33】
{
    MkiOpTest opTest;
    opTest.Golden(std::bind(&Golden::InOutTensorEqual, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_FLOAT16;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase2) // FLOAT32 -> INT32 【cast_float_to_int32; //0010 0100  36】
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGolden2Int32, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT32;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase3) // FLOAT16 -> INT32 【cast_half_to_int32; //0001  0101  21 】
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGolden2Int32, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT32;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase4) // INT64 -> INT32 【cast_int64_to_int32; //0111 0010  114】
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGoldenINT, ATOL, RTOL, std::placeholders::_1));
    opTest.LongRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT32;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1000}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase5) // INT32 -> INT64 cast_int32_to_int64; //0110 0011  99】
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGoldenINT, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT64;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {1000}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase6) // INT32 -> INT64 cast_int32_to_int64; //0110 0011  99】
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGoldenINT, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT64;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {129}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase7) // INT32 -> INT64 cast_int32_to_int64; //0110 0011  99】
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGoldenINT, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT64;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {127}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase8) // INT32 -> INT64 cast_int32_to_int64; //0110 0011  99】
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGoldenINT, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT64;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {8}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase9) // INT32 -> INT64 cast_int32_to_int64; //0110 0011  99】
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGoldenINT, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT64;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {4}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase10) // INT32 -> INT64 cast_int32_to_int64; //0110 0011  99】
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGoldenINT, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT64;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {11}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCast, CastWideCase11) // INT32 -> INT64 cast_int32_to_int64; //0110 0011  99】
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    MkiOpTest opTest;
    opTest.Golden(std::bind(&CastWideGoldenINT, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_CAST};
    opParam.outTensorType = TENSOR_DTYPE_INT64;
    UtOpDesc opDesc = {"ElewiseOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {12}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}