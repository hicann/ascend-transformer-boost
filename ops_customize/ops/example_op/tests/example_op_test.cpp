#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "example_operation.h"
#include "customize_op_params.h"
#include "atb/core/op_param_funcs.h"
#include "atb/utils.h"
#include "atb/context.h"

using namespace atb;

const int32_t DEVICE_ID = 0;
const int VECTOR_SIZE = 4;
const float INIT_VALUE = 2.0f;

#define CHECK_STATUS(status)                                                                  \
    do {                                                                                      \
        if ((status) != 0) {                                                                  \
            std::cout << __FILE__ << ": " <<__LINE__ << "[error]: " << (status) << std::endl; \
            exit(1);                                                                          \
        }                                                                                     \
    } while(0)

atb::Tensor CreateTensor(const aclDataType dataType, const aclFormat format, std::vector<int64_t> shape)
{
    atb::Tensor tensor;
    tensor.desc.dtype = dataType;
    tensor.desc.format = format;
    tensor.desc.shape.dimNum = shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
        tensor.desc.shape.dims[i] = shape.at(i);
    }
    tensor.dataSize = atb::Utils::GetTensorSize(tensor);
    CHECK_STATUS(aclrtMalloc(&tensor.deviceData, tensor.dataSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
    return tensor;
}

template <typename T>
atb::Tensor CreateTensorFromVector(std::vector<T> data, const aclDataType outTensorType, const aclFormat format,
                                   std::vector<int64_t> shape)
{
    atb::Tensor tensor;
    tensor = CreateTensor(outTensorType, format, shape);
    CHECK_STATUS(aclrtMemcpy(
        tensor.deviceData, tensor.dataSize, data.data(), sizeof(T) * data.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    return tensor;
}

atb::SVector<atb::Tensor> PrepareAddInTensors()
{
    uint32_t dim0 = 2;
    uint32_t dim1 = 2;
    std::vector<__fp16> tensor0(VECTOR_SIZE, static_cast<__fp16>(INIT_VALUE));
    atb::Tensor tensorAdd0 = 
        CreateTensorFromVector(tensor0, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {dim0, dim1});
    std::vector<__fp16> tensor1(VECTOR_SIZE, static_cast<__fp16>(INIT_VALUE));
    atb::Tensor tensorAdd1 = 
        CreateTensorFromVector(tensor1, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {dim0, dim1});
    atb::SVector<atb::Tensor> inTensors = {tensorAdd0, tensorAdd1};
    return inTensors;
}

atb::Operation *PrepareOperation()
{
    atb::customize::AddParam param;
    param.paramForNothing = 0;
    atb::Operation *op = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &op));
    return op;
}

// TEST(ExampleOpTest, CreateOperation_Success) {
//     customize::AddParam param;
//     param.paramForNothing = 0;

//     Operation *op = nullptr;
//     Status st = CreateOperation(param, &op);
//     EXPECT_EQ(st, NO_ERROR) << "Create op should return NO_ERROR";
//     ASSERT_NE(op, nullptr) << "op can not be nullptr";

//     delete op;
// }

TEST(ExampleOpTest, CreateOperation_Success) {
    atb::Context *context = nullptr;
    void *stream = nullptr;

    CHECK_STATUS(aclInit(nullptr));
    CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
    CHECK_STATUS(CreateContext(&context));
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    atb::Operation *op = PrepareOperation();
    atb::VariantPack variantPack;
    variantPack.inTensors = PrepareAddInTensors();
    atb::Tensor tensorOut = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {2, 2});
    variantPack.outTensors.push_back(tensorOut);

    // setup
    uint64_t workspaceSize = 0;
    CHECK_STATUS(op->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    //execute
    op->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream));

    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(op));
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context));
    CHECK_STATUS(aclFinalize());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}