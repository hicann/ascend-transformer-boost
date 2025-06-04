#include "c_interface_utils.h"
#include "acl/acl.h"
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
    for (size_t i = 0; i < num; i++) {
        aclrtMallocHost((void **)(&(inoutHost[i])), inoutSize[i]);
        aclrtMalloc((void **)(&(inoutDevice[i])), inoutSize[i], ACL_MEM_MALLOC_HUGE_FIRST);
    }
    return ACL_ERROR_NONE;
}

void CreateACLTensorInOut(const std::vector<int64_t> dims, aclDataType type, aclFormat format, aclTensor **list,
                          size_t &i, void *inout)
{
    list[i++] = aclCreateTensor(dims.data(), dims.size(), type, nullptr, 0, format, nullptr, 0, inout);
}

uint64_t GetDataTypeSize(const aclDataType &dType)
{
    switch (dType) {
        case ACL_DT_UNDEFINED:
        case ACL_BOOL:
            return sizeof(bool);
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_UINT8:
            return sizeof(uint8_t);
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_INT16:
            return sizeof(int16_t);
        case ACL_UINT16:
            return sizeof(uint16_t);
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_UINT32:
            return sizeof(uint32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        case ACL_UINT64:
            return sizeof(uint64_t);
        case ACL_DOUBLE:
            return sizeof(double);
        case ACL_STRING:
            return sizeof(std::string);
        case ACL_COMPLEX64:
            return sizeof(std::complex<float>);
        case ACL_COMPLEX128:
            return sizeof(std::complex<double>);
        default:
            return 0;
    }
    return 0;
}

}
}