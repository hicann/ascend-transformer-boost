#include "c_interface_utils.h"
namespace atb {
namespace cinterfaceTest {
int64_t GetTensorSize(const aclTensor *input)
{
    const op::Shape shape = input->GetViewShape();
    const size_t dims = shape.GetDimNum();
    int64_t size = 1;
    for (size_t i = 0; i < dims; ++i) {
        size *= shape.GetDim(i);
    }
    return size;
}

aclnnStatus Init(atb::Context **context, aclrtStream *stream, int64_t *deviceId)
{
    aclInit(nullptr);
    aclrtSetDevice(*deviceId);
    aclrtCreateStream(stream);
    atb::CreateContext(context);
    (*context)->SetExecuteStream(*stream);
    return ACL_ERROR_NONE;
}

aclnnStatus Destroy(atb::Context **context, aclrtStream *stream)
{
    aclrtDestroyStream(*stream);
    DestroyContext(*context);
    aclFinalize();
    return ACL_ERROR_NONE;
}

aclnnStatus CreateInOutData(size_t num, uint8_t **inoutHost, uint8_t **inoutDevice, size_t *inoutSize)
{
    for(size_t i = 0; i < num; i++) {
        aclrtMallocHost((void **)(&(inoutHost[i])), inoutSize[i]);
        aclrtMalloc((void **)(&(inoutDevice[i])), inoutSize[i], ACL_MEM_MALLOC_HUGE_FIRST);
    }
    return ACL_ERROR_NONE;
}

void CreateACLTensorInOut(const std::vector<int64_t> dims, aclDataType type, aclFormat format, aclTensor **list, size_t &i, uint8_t *inout)
{
    list[i++] = aclCreateTensor(dims.data(), dims.size(), type, nullptr, 0, format, nullptr, 0, inout);
}
}
}