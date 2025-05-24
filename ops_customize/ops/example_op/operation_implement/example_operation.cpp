#include "example_operation.h"
#include "example_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/config.h"
#include "atb/utils/param_to_json.h"
#include "atb/core/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/core/op_param_funcs.h"

namespace atb {
const uint32_t TENSOR_NUM_ONE = 1;
const uint32_t TENSOR_NUM_TWO = 2;
const uint32_t TENSOR_NUM_THREE = 3;
const uint32_t TENSOR_IDX_ZERO = 0;
const uint32_t TENSOR_IDX_ONE = 1;
const uint32_t TENSOR_IDX_TWO = 2;

template <> Status CreateOperation(const customize::AddParam &opParam, Operation **operation)
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

uint32_t AddOperation::GetInputNum() const
{
    return TENSOR_NUM_TWO;
}

uint32_t AddOperation::GetOutputNum() const
{
    return TENSOR_NUM_ONE;
}

Status AddOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status AddOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.size() != 2) {
        ATB_LOG(ERROR) << "AddOp requires exactly 2 inputs, got ";
                       << inTensorDescs.size();
        return ERROR_INVALID_PARAM;
    }

    const TensorDesc &A = inTensorDescs[0];
    const TensorDesc &B = inTensorDescs[1];

    if (A.dtype != B.dtype || A.dtype != ACL_FLOAT16) {
        ATB_LOG(ERROR) << "AddOp inputs dtypes must match and be FLOAT16, got ";
                       << A.dtype << " vs " <<B.dtype;
        return ERROR_INVALID_PARAM;
    }

      if (A.format != B.format) {
        ATB_LOG(ERROR) << "AddOp input formats must match, got "
                       << A.format << " vs " << B.format;
        return ERROR_INVALID_PARAM;
    }

    if (A.shape.dimNum == 0 || A.shape.dimNum > MAX_DIM ||
        B.shape.dimNum == 0 || B.shape.dimNum > MAX_DIM) {
        ATB_LOG(ERROR) << "AddOp input dimNum out of range, got "
                       << A.shape.dimNum << " and " << B.shape.dimNum;
        return ERROR_INVALID_PARAM;
    }

    for (uint64_t i = 0; i < A.shape.dimNum; ++i) {
        if (A.shape.dims[i] <= 0) {
            ATB_LOG(ERROR) << "AddOp input 0 dim[" << i << "] <= 0";
            return ERROR_INVALID_PARAM;
        }
    }
    for (uint64_t i = 0; i < B.shape.dimNum; ++i) {
        if (B.shape.dims[i] <= 0) {
            ATB_LOG(ERROR) << "AddOp input 1 dim[" << i << "] <= 0";
            return ERROR_INVALID_PARAM;
        }
    }
    return NO_ERROR;
}

Status AddOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (inTensors.size() != 2 || outTensors.size() != 1) {
        ATB_LOG(ERROR) << "AddOp runtime expects 2 inputs & 1 output, got "
                       << inTensors.size() << " in, "
                       << outTensors.size() << " out";
        return ERROR_INVALID_PARAM;
    }

    const TensorDesc &TA = inTensors[0].desc;
    const TensorDesc &TB = inTensors[1].desc;
    const TensorDesc &TC = outTensors[0].desc;

    if (TA.dtype != TB.dtype || TA.dtype != TC.dtype || TA.dtype != ACL_FLOAT16) {
        ATB_LOG(ERROR) << "AddOp runtime dtype mismatch or unsupported.";
        return ERROR_INVALID_PARAM;
    }
    if (TA.format != TB.format || TA.format != TC.format) {
        ATB_LOG(ERROR) << "AddOp runtime format mismatch.";
        return ERROR_INVALID_PARAM;
    }
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    return NO_ERROR;
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