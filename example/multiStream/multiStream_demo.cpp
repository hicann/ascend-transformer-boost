#include "atb/atb_infer.h"
#include <acl/acl.h>
#include <iostream>
#include <vector>
#include <atb/utils.h>
#include <unistd.h>
#include <acl/acl_mdl.h>
#include <cstdlib>  // 包含 std::getenv
#include <cstring>  // 包含 std::strcmp（用于字符串比较）

static void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs) 
{
    for (size_t i = 0; i < intensorDescs.size(); i++) {
        intensorDescs.at(i).dtype = ACL_FLOAT16;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = 2;
        intensorDescs.at(i).shape.dims[0] = 2;
        intensorDescs.at(i).shape.dims[1] = 2;
    }
}

static void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
{
    std::vector<char> zeroData(8, 0); // 一段全0的hostBuffer
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        int ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
        ret = aclrtMemcpy(inTensors.at(i).deviceData, inTensors.at(i).dataSize, zeroData.data(), zeroData.size(), ACL_MEMCPY_HOST_TO_DEVICE); //拷贝CPU内存到NPU侧
    }
}

// 设置各个outtensor并且为outtensor分配内存空间，同intensor设置
static void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++) {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
}

static void CreateMiniGraphOperation(atb::GraphParam &opGraph, atb::Operation **operation)
{
	// 构子图流程
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(3);

    size_t nodeId = 0;
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &addNode3 = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam, &addNode.operation);
    addNode.inTensorIds = {0, 1};
    addNode.outTensorIds = {3};

    atb::infer::ElewiseParam addParam2;
    addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam2, &addNode2.operation);
    addNode2.inTensorIds = {3, 1};
    addNode2.outTensorIds = {4};

    atb::infer::ElewiseParam addParam3;
    addParam3.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam3, &addNode3.operation);
    addNode3.inTensorIds = {4, 1};
    addNode3.outTensorIds = {2};

    atb::CreateOperation(opGraph, operation);
}

static void CreateGraphOperationForMultiStream(atb::GraphParam &opGraph, atb::Operation **operation)
{
    // 构单图多流大图
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 2;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(4);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &graphNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulNode1 = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mulParam, &mulNode.operation);
    mulNode.inTensorIds = {0, 1};
    mulNode.outTensorIds = {3};

    atb::infer::ElewiseParam addParam2;
    addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam2, &addNode2.operation);
    addNode2.inTensorIds = {0, 1};
    addNode2.outTensorIds = {4};

    atb::GraphParam graphParam;
    CreateMiniGraphOperation(graphParam, &graphNode.operation);
    graphNode.inTensorIds = {4, 1};
    graphNode.outTensorIds = {5};
    SetExecuteStreamId(graphNode.operation, 1);

    atb::CreateOperation(mulParam, &mulNode1.operation);
    mulNode1.inTensorIds = {5, 1};
    mulNode1.outTensorIds = {2};
    SetExecuteStreamId(mulNode1.operation, 1);

    atb::CreateOperation(opGraph, operation);
}

static void CreateGraphOperationWithEvent(atb::GraphParam &opGraph, atb::Operation **operation, aclrtEvent event)
{
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(5);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &waitNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &graphNode = opGraph.nodes.at(nodeId++);
    atb::Node &recordNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mulParam, &mulNode.operation);
    mulNode.inTensorIds = {0, 1};
    mulNode.outTensorIds = {3};

    atb::common::EventParam waitParam;
    waitParam.event = event;
    waitParam.operatorType = atb::common::EventParam::OperatorType::WAIT;
    atb::CreateOperation(waitParam, &waitNode.operation);

    atb::GraphParam graphParam;
    CreateMiniGraphOperation(graphParam, &graphNode.operation);
    graphNode.inTensorIds = {3, 4};
    graphNode.outTensorIds = {2};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam, &addNode.operation);
    addNode.inTensorIds = {0, 1};
    addNode.outTensorIds = {4};
    SetExecuteStreamId(addNode.operation, 1);

    atb::common::EventParam recordParam;
    recordParam.event = event;
    recordParam.operatorType = atb::common::EventParam::OperatorType::RECORD;
    atb::CreateOperation(recordParam, &recordNode.operation);
    SetExecuteStreamId(recordNode.operation, 1);

    atb::CreateOperation(opGraph, operation);
}

// 打印结果
void PrintOutTensorValue(atb::Tensor &outTensor)
{
    std::vector<uint16_t> outBuffer(atb::Utils::GetTensorNumel(outTensor));
    int ret = aclrtMemcpy(outBuffer.data(), outBuffer.size() * sizeof(uint16_t), outTensor.deviceData, outTensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != 0) {
        std::cout << "copy error!";
        exit(0);
    }
    for (size_t i = 0; i < outBuffer.size(); i = i + 1) {
        std::cout << "out[" << i << "] = " << (uint32_t)outBuffer.at(i) << std::endl;
    }
}

int main()
{
    aclInit(nullptr);
    // 使用环境变量ATB_MULTISTREAM_TYPE判断使用单图多流还是多图多流功能
    // ATB_MULTISTREAM_TYPE=singleGraph ：单图多流
    // ATB_MULTISTREAM_TYPE=MultiGraph ：多图多流
    const char* multiStreamType = std::getenv("ATB_MULTISTREAM_TYPE");
	// 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    // 创建多个stream
    aclrtStream stream1 = nullptr;
    aclrtCreateStream(&stream1);
    aclrtStream stream2 = nullptr;
    aclrtCreateStream(&stream2);
    std::vector<aclrtStream> streams = {stream1, stream2};

    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStreams(streams);
    // 创建图算子
    atb::Operation *operation = nullptr;
    atb::GraphParam opGraph;
    CreateGraphOperationForMultiStream(opGraph, &operation);

	// 输入输出tensor准备
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;

    uint32_t inTensorNum = opGraph.inTensorNum;
    uint32_t outTensorNum = opGraph.outTensorNum;
    inTensorNum = operation->GetInputNum();
    outTensorNum = operation->GetOutputNum();

    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);
    
    CreateInTensorDescs(intensorDescs);
    
    outtensorDescs.resize(outTensorNum);
    pack.outTensors.resize(outTensorNum);
    operation->InferShape(intensorDescs, outtensorDescs);
    CreateOutTensors(pack.outTensors, outtensorDescs);

    // 初始化workspace
    uint64_t workspaceSize = 0;
    void *workSpace = nullptr;
    int ret = 0;
    // 图执行
    if (!multiStreamType || std::strcmp(multiStreamType, "singleGraph") == 0) {
        // 图内多流并行
        std::cout << "Single graph multi-stream demo start" << std::endl;
        CreateInTensors(pack.inTensors, intensorDescs);
        operation->Setup(pack, workspaceSize, context);
        if (workspaceSize != 0 && workSpace == nullptr) {
            ret = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != 0) {
                std::cout << "alloc error!\n";
                exit(0);
            }
        }
        operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
        //流同步
        ret = aclrtSynchronizeStream(stream1);
        if (ret != 0) {
            std::cout << "sync error!";
            exit(0);
        }

        ret = aclrtSynchronizeStream(stream2);
        if (ret != 0) {
            std::cout << "sync error!";
            exit(0);
        }
    } else {
        // 图间同步
        std::cout << "multi graph multi-stream demo start" << std::endl;
    }

	// 资源释放
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtDestroyStream(stream1);
    aclrtDestroyStream(stream2);
    aclrtResetDevice(deviceId);
    aclFinalize();
}
