#include "utils/log.h"
#include "utils/utils.h"

void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs)
{
    for (size_t i = 0; i < intensorDescs.size(); i++)
    {
        intensorDescs.at(i).dtype = ACL_FLOAT16;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = 2;
        intensorDescs.at(i).shape.dims[0] = 2;
        intensorDescs.at(i).shape.dims[1] = 2;
    }
}

void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
{
    for (size_t i = 0; i < inTensors.size(); i++)
    {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        std::vector<uint16_t> hostData(atb::Utils::GetTensorNumel(inTensors.at(i)), 2); // 一段全2的hostBuffer
        int ret = aclrtMalloc(
            &inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        CHECK_RET(ret, "alloc error!");

        ret = aclrtMemcpy(inTensors.at(i).deviceData,
                          inTensors.at(i).dataSize,
                          hostData.data(),
                          hostData.size() * sizeof(uint16_t),
                          ACL_MEMCPY_HOST_TO_DEVICE); // 拷贝CPU内存到NPU侧
        CHECK_RET(ret, "aclrtMemcpy error!");
    }
}

void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++)
    {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret, "aclrtMalloc error!");
    }
}

void CreateTensorFromDesc(atb::Tensor &tensor, atb::TensorDesc &tensorDescs)
{
    tensor.desc = tensorDescs;
    tensor.dataSize = atb::Utils::GetTensorSize(tensor);
    int ret = aclrtMalloc(&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret, "aclrtMalloc error!");
}

void PrintOutTensorValue(atb::Tensor &outTensor)
{
    // 输出tensor拷贝回host侧并打印
    std::vector<uint16_t> outBuffer(atb::Utils::GetTensorNumel(outTensor));
    int ret = aclrtMemcpy(outBuffer.data(),
                          outBuffer.size() * sizeof(uint16_t),
                          outTensor.deviceData,
                          outTensor.dataSize,
                          ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret, "copy error!");

    for (size_t i = 0; i < outBuffer.size(); i = i + 1)
    {
        LOG_INFO("out[" + std::to_string(i) + "] = " + std::to_string((uint32_t)outBuffer.at(i)));
    }
}

