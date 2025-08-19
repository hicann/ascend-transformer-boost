#include <gtest/gtest.h>
#include <torch/torch.h>
#include <atb/utils/log.h>
#include <atb/utils.h>
#include "test_utils/test_common.h"
#include "atb/operation.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "atb/utils/operation_util.h"
#include "atb/operation/operation_base.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

using namespace atb;

TEST(TestRepeatInplaceWrite, graphTest)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        GTEST_SKIP() << "This test case only support 910B";
    }

    int8_t worldSize = 2;

    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = =7;
    opGraph.outTensorNum = 3;
    opGraph.internalTensorNum = 8;
    opGraph.nodes.resize(8);

    size_t nodeId = 0;
    atb::Node &node0 = opGraph.nodes.at(nodeId++);
    atb::Node &node1 = opGraph.nodes.at(nodeId++);
    atb::Node &node2 = opGraph.nodes.at(nodeId++);
    atb::Node &node3 = opGraph.nodes.at(nodeId++);
    atb::Node &node4 = opGraph.nodes.at(nodeId++);
    atb::Node &node5 = opGraph.nodes.at(nodeId++);
    atb::Node &node6 = opGraph.nodes.at(nodeId++);
    atb::Node &node7 = opGraph.nodes.at(nodeId++);
    
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    linearParam.transposeB = true;
    atb::Status status1 = atb::CreateOperation(linearParam, &node0.operation);
    EXPECT_EQ(status1, 0);
    node0.inTensorIds = {0, 1};
    node0.outTensorIds = {10};

    atb::infer::AllReduceParam allReduceParam;
    allreduceParam.rank = 1;
    allreduceParam.rankSize = worldSize;
    allreduceParam.backend = "lccl";
    allreduceParam.allReduceType = "sum";
    allreduceParam.rankTableFile = "";
    atb::Status status2 = CreateOperation(allReduceParam, &node1.operation);
    EXPECT_EQ(status2, 0);
    node1.inTensorIds = {10};
    node1.outTensorIds = {11};

    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.epsilon = 1e-06;
    atb::Status status3 = CreateOperation(rmsNormParam, &node2.operation);
    EXPECT_EQ(status3, 0);
    node2.inTensorIds = {11, 2, 3};
    node2.outTensorIds = {12, 7, 13};

    // reuse linear param
    atb::Status status4 = atb::CreateOperation(linearParam, &node3.operation);
    EXPECT_EQ(status4, 0);
    node3.inTensorIds = {12, 4};
    node3.outTensorIds = {14};

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = ActivationType::ACTIVATION_SWIGLU_FORWARD;
    activationParam.scale = 1.0f;
    activationParam.dim = -1;
    activationParam.geluMode = 0; //TANH_MODE
    atb::Status status5 = atb::CreateOperation(activationParam, &node4.operation);
    EXPECT_EQ(status5, 0);
    node4.inTensorIds = {14};
    node4.outTensorIds = {15};

    // reuse linear param
    atb::Status status6 = atb::CreateOperation(linearParam, &node5.operation);
    EXPECT_EQ(status6, 0);
    node5.inTensorIds = {15, 5};
    node5.outTensorIds = {16};

    // reuse allreduce param
    atb::Status status7 = CreateOperation(allReduceParam, &node6.operation);
    EXPECT_EQ(status7, 0);
    node6.inTensorIds = {16};
    node6.outTensorIds = {17};

    // reuse rmsnorm param
    atb::Status status8 = CreateOperation(rmsNormParam, &node7.operation);
    EXPECT_EQ(status8, 0);
    node7.inTensorIds = {17, 13, 6};
    node7.outTensorIds = {8, 7, 9};

    atb::Status status9 = atb::CreateOperation(opGraph, &operation);
    EXPECT_EQ(status9, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status10 = opTest.Run(operation, inTensorDescs);
    DestroyOperation(operation);
    EXPECT_EQ(status10, 0);
}