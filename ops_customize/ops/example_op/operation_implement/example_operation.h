#ifndef CUSTOMIZE_ADDOPERATION_H
#define CUSTOMIZE_ADDOPERATION_H
#include <acl/acl.h>
#include "atb/operation/operation_base.h"
#include "customize_op_params.h"

namespace atb {
class AddOperation : public OperationBase {
public:
    explicit AddOperation(const customize::AddParam &param);
    ~AddOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status AddOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const;
    Status AddOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    nlohmann::json GetParamJson() const override;

private:
    customize::AddParam param_;
};
} // namespace atb
#endif