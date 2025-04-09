import torch
import torch.nn as nn
import torch_atb  
from utils import check_float, run_perf_test

def run_test():
    print("----------- self_attention test begin ------------")
    self_attention_param = torch_atb.SelfAttentionParam()
    self_attention_param.head_num = 24
    self_attention_param.kv_head_num = 24
    self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
    self_attention = torch_atb.Operation(self_attention_param)
    q = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    k = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    v = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    seqlen = torch.tensor([4096], dtype=torch.int32)
    intensors = [q,k,v,seqlen]

    def self_attention_run():
        outputs = self_attention.forward([q,k,v,seqlen])
        return []

    npu_outputs = self_attention_run()
    print("npu_outputs: ", npu_outputs)
    run_perf_test(self_attention, intensors)
    print("----------- self_attention test success ------------")

if __name__ == "__main__":
    run_test()