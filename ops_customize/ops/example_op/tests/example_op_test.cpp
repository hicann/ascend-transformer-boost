#include <gtest/gtest.h>
#include "example_operation.h"
#include "customize_op_params.h"
#include "atb/core/op_param_funcs.h"

using namespace atb;

TEST(ExampleOpTest, CreateOperation_Success) {
    customize::AddParam param;
    param.paramForNothing = 0;

    Operation *op = nullptr;
    Status st = CreateOperation(param, &op);
    EXPECT_EQ(st, NO_ERROR) << "Create op should return NO_ERROR";
    ASSERT_NE(op, nullptr) << "op can not be nullptr";

    delete op;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}