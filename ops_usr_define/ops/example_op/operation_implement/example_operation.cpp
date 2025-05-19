#include "example_operation.h"
#include "example_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/config.h"
#include "atb/utils/param_to_json.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/core/op_param_funcs.h"

namespace atb {
template <> Status CreateOperation(const usr_define::AddParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.paramForNothing != 0) {
        ATB_LOG(ERROR) << "param: paramForNothing should be 0";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) AddOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

AddOperation::AddOperation(const customize::AddParam &param) : OperationBase("AddOperation"), param_(param) {}

AddOperation::~AddOperation() {}

Status AddOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

SVector<bool> AddOperation::GetEmptyInTensorPermissions() const
{
    SVector<bool> v;

    // quant_per_channel dequant_per_channel intensor[2] allows null
    if (GetInputNum() == TENSOR_NUM_THREE) {
        SVector<bool> emptyTensorPerms(GetInputNum(), false);
        emptyTensorPerms.at(TENSOR_NUM_THREE - 1) = true;
        return emptyTensorPerms;
    }

    return v;
}

std::shared_ptr<Runner> AddOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<AddOpsRunner>(param_);
}

nlohmann::json AddOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb