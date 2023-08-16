#include <gtest/gtest.h>
#include <torch/torch.h>
#include <asdops/utils/log/log.h>
#include "tests/unittest/test_util/test_common.h"
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include "tests/unittest/test_util/op_test.h"
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
using namespace AclTransformer;
using namespace AsdOps;

TEST(TestRopeOperation, InferShape)
{
    AclTransformer::PositionEmbeddingFusionParam param;
    AclTransformer::RopeOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensors = {{AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 1, 1}},
                                                 {AsdOps::TENSOR_DTYPE_FLOAT16, AsdOps::TENSOR_FORMAT_ND, {2, 2, 2, 2}},
                                                 {AsdOps::TENSOR_DTYPE_INT64, AsdOps::TENSOR_FORMAT_ND, {3, 3, 3, 3}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensors, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    EXPECT_EQ(outTensorDescs.at(0).format, AsdOps::TENSOR_FORMAT_ND);
    AsdOps::SVector<int64_t> expectDims = {1, 1, 0, 0};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(0).dims.at(3));

    EXPECT_EQ(outTensorDescs.at(1).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    EXPECT_EQ(outTensorDescs.at(1).format, AsdOps::TENSOR_FORMAT_ND);
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(1).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(1).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(1).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(1).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(1).dims.at(3));

    EXPECT_EQ(outTensorDescs.at(2).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    EXPECT_EQ(outTensorDescs.at(2).format, AsdOps::TENSOR_FORMAT_ND);
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(2).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(2).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(2).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(2).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(2).dims.at(3));
    return AsdOps::Status::OkStatus();
}

TEST(TestRopeOperation, InferShape2)
{
    AclTransformer::PositionEmbeddingFusionParam param;
    param.headNum = 4;
    param.model = "chatglm16";
    AclTransformer::RopeOperation op(param);
    AsdOps::SVector<AsdOps::Tensor> inTensors = {{AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 36, 1}},
                                                 {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 8, 8}},
                                                 {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 8, 8}},
                                                 {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 8, 8}},
                                                 {AsdOps::TENSOR_DTYPE_FLOAT, AsdOps::TENSOR_FORMAT_ND, {1, 1, 8, 8}}};
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    op.InferShape(inTensors, outTensorDescs);
    ASSERT_EQ(outTensorDescs.size(), 3);
    EXPECT_EQ(outTensorDescs.at(0).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    EXPECT_EQ(outTensorDescs.at(0).format, AsdOps::TENSOR_FORMAT_ND);
    AsdOps::SVector<int64_t> expectDims = {1, 1, 4, 3};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(0).dims.at(3));

    EXPECT_EQ(outTensorDescs.at(1).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    EXPECT_EQ(outTensorDescs.at(1).format, AsdOps::TENSOR_FORMAT_ND);
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(1).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(1).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(1).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(1).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(1).dims.at(3));

    EXPECT_EQ(outTensorDescs.at(1).dtype, AsdOps::TENSOR_DTYPE_FLOAT);
    EXPECT_EQ(outTensorDescs.at(1).format, AsdOps::TENSOR_FORMAT_ND);
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(2).dims.size());
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(2).dims.at(0));
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(2).dims.at(1));
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(2).dims.at(2));
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(2).dims.at(3));

    return AsdOps::Status::OkStatus();
}