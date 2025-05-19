#include <iostream>
#include "atb/utils/log.h"
#include "example_ops_runner.h"

namespace atb {
static const uint32_t NUMONE = 1;
static const uint32_t NUMTWO = 2;
static const uint32_t NUMTHREE = 3;
static const uint32_t INDEX_ZERO = 0;
static const uint32_t INDEX_ONE = 1;
static const uint32_t INDEX_TWO = 2;

AddOpsRunner::AddOpsRunner(const customize::AddParam &param)
    : OpsRunner("AddOpsRunner", RUNNER_TYPE_ELEWISE), param_(param)
{
    ATB_LOG(INFO) << "AddOpsRunner::AddOpsRunner called";

    kernelGraph_.nodes.resize(NUMONE);
    auto &addNode = kernelGraph_.nodes.at(INDEX_ZERO);

    Mki::Tensor &aTensor = kernelGraph_.inTensors.at(INDEX_ZERO);
    Mki::Tensor &bTensor = kernelGraph_.inTensors.at(INDEX_ONE);
    addNode.inTensors = {&aTensor, &bTensor};

    kernelGraph_.outTensors.resize(NUMONE);
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(INDEX_ZERO);
    addNode.outTensors = {&operationOutTensor};

    AsdOps::OpParam::Elewise AddParam{ AsdOps::OpParam::Elewise::ELEWISE_ADD }
    addNode.opDesc = {0, "AddOperation", AddParam};
    ATB_LOG(INFO) << "AddOpsRunner::AddOpsRunner end";
}

AddOpsRunner::~AddOpsRunner() {}

} // namespace atb