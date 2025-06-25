import torch
import torch_atb
import torch_npu
import acl
import unittest

device_id = 7
class topktopp:
    def __init__(self):
        self.param = torch_atb.TopkToppSamplingParam()
        self.param.topk_topp_sampling_type = torch_atb.TopkToppSamplingParam.TopkToppSamplingType.BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING
        self.param.log_probs_size = 15
        self.op = torch_atb.Operation(self.param)
    
    def forward(self, tensor: list[torch.tensor]) -> list[torch.tensor]:
        return self.op.forward(tensor)

def gen_inputs():
    t1 = (torch.rand(4, 50, dtype=torch.float16) * 200.0 - 100.0).npu()
    t2 = torch.randint(low=10, high=21, size=(4, 1), dtype=torch.int32).npu()
    t3 = torch.randn(4, 1, dtype=torch.float16).npu()
    t4 = torch.randn(4, 1, dtype=torch.float).npu()

    return [t1, t2, t3, t4]

def run_test():
    print("----------- topk_topp_sampling test begin ------------")
    topktopsampling = topktopp()
    topktopsampling.forward(gen_inputs())
    print("----------- topk_topp_sampling test success ------------")
    
class TestTopkToppSampling(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()