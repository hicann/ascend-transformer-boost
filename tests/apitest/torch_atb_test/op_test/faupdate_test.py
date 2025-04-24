import torch
import torch_atb
import re
from utils import run_perf_test


def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10) or re.search("Ascend910_93", device_name, re.I)


def run_test():
    if not is910B():
        print("This test case only support 910B")
        return
    print("----------- faupdate test begin ------------")
    faupdate_param = torch_atb.FaUpdateParam()
    faupdate_param.fa_update_type = torch_atb.FaUpdateParam.FaUpdateType.DECODE_UPDATE
    faupdate_param.sp = 1
    faupdate = torch_atb.Operation(faupdate_param)

    hDim = 8
    b = 2
    s = 1
    sp = 1
    hc = 1
    shape0 = (sp, b*s*hc)
    shape1 = (sp, b*s*hc, hDim)

    #input0: lse   input1: go
    input0 = torch.rand(shape0, dtype=torch.float32).npu()
    input1 = torch.rand(shape1, dtype=torch.float32).npu()

    intensors = [input0, input1]
    run_perf_test(faupdate, intensors, 10)
    print("----------- faupdate test success ------------")


if __name__ == "__main__":
    run_test()
