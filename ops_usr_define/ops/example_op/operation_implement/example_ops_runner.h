#ifndef CUSTOMIZE_ADDOPSRUNNER_H
#define CUSTOMIZE_ADDOPSRUNNER_H
#include <asdops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "customize_op_params.h"

namespace atb {
class AddOpsRunner : public OpsRunner {
public:
    explicit AddOpsRunner(const customize::AddOParam &param);
    ~AddOpsRunner() override;

private:
    customize::AddParam param_;
    // Mki::Tensor nullTensor_ = {}; // 空tensor占位符
};
} // namespace atb
#endif