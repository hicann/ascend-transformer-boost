#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "customize_block_copy_operation.h"
#include "customize_op_params.h"
#include "atb/core/op_param_funcs.h"
#include "atb/utils.h"
#include "atb/context.h"

using namespace atb;

const int32_t DEVICE_ID = 0;
const int BLOCK_COUNT = 2;
const int BLOCK_SIZE = 2;
const int HEADS = 1;
const int HEADS_SIZE = 1;
const float INIT_VALUE = 1.0f;

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

atb::SVector<atb::Tensor> PrepareBlockCopyInTensors()
{
    int total = BLOCK_COUNT * BLOCK_SIZE;
    std::vector<__fp16> hostKV(total);
    for (int i = 0; i < total; ++i) {
        hostKV[i] = static_cast<__fp16>(INIT_VALUE + i);
    }
    auto keyCache   = CreateTensorFromVector(hostKV, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {BLOCK_COUNT, BLOCK_SIZE, HEADS, HEAD_SIZE});
    auto valueCache = CreateTensorFromVector(hostKV, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {BLOCK_COUNT, BLOCK_SIZE, HEADS, HEAD_SIZE});

    // 2) srcBlockIndices, cumSum：长度 BLOCK_COUNT = 2，cumSum = [1,2]
    std::vector<int32_t> srcIdx = {0, 1};
    std::vector<int32_t> cumSum = {1, 2};
    auto srcTensor = CreateTensorFromVector(srcIdx, ACL_INT32, aclFormat::ACL_FORMAT_ND, {BLOCK_COUNT});
    auto cumSumTensor = CreateTensorFromVector(cumSum, ACL_INT32, aclFormat::ACL_FORMAT_ND, {BLOCK_COUNT});

    // 3) dstBlockIndices：长度 cumSum.back() = 2，对应 [1,0] 表示交换两个块
    std::vector<int32_t> dstIdx = {1, 0};
    auto dstTensor = CreateTensorFromVector(dstIdx, ACL_INT32, aclFormat::ACL_FORMAT_ND, {static_cast<int64_t>(cumSum.back())});

    return { keyCache, valueCache, srcTensor, dstTensor, cumSumTensor };
}

atb::Operation *PrepareOperation()
{
    atb::customize::BlockCopyParam param;
    atb::Operation *op = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &op));
    return op;
}

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
    variantPack.inTensors = PrepareBlockCopyInTensors();
    auto outKey   = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {BLOCK_COUNT, BLOCK_SIZE, HEADS, HEAD_SIZE});
    auto outValue = CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {BLOCK_COUNT, BLOCK_SIZE, HEADS, HEAD_SIZE});
    pack.outTensors.push_back(outKey);
    pack.outTensors.push_back(outValue);

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