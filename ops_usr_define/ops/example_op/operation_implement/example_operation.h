#ifndef USR_DEFINE_ADDOPERATION_H
#define USR_DEFINE_ADDOPERATION_H
#include <acl/acl.h>
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

// copy from elewise
namespace atb {
class AddOperation : public OperationBase {
public:
    explicit AddOperation(const usr_define::AddOperation &param);
    ~AddOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status InferShapeImplCast(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImplQuant(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImplDynamicQuant(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    SVector<bool> GetEmptyInTensorPermissions() const override;
    nlohmann::json GetParamJson() const override;

private:
    usr_define::AddOperation param_;
    Status InferShapeCommon(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImplQuantChannel(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImplDequantChannel(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const;
    bool InferShapeCheckDynamicQuant(const SVector<TensorDesc> &inTensorDescs) const;
    bool InferShapeCheckDynamicQuant310P(const SVector<TensorDesc> &inTensorDescs) const;
};
} // namespace atb
#endif