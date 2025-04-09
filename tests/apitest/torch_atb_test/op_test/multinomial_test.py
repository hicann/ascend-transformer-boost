import torch
import torch_atb  
from utils import run_perf_test

def run_test():
    print("----------- multinomial test begin ------------")
    rand_seed = 123
    intensor = torch.rand(3, 3, dtype=torch.float16)
    normalized_tensor = intensor / intensor.sum()
    normalized_tensor_npu = normalized_tensor.npu()
    multinomial_param = torch_atb.MultinomialParam(rand_seed = rand_seed)
    multinomial = torch_atb.Operation(multinomial_param)
    print(multinomial_param)

    def multinomial_run():
        multinomial_outputs = multinomial.forward([normalized_tensor_npu])
        return multinomial_outputs

    npu_outputs = multinomial_run()
    print("npu_outputs: ", npu_outputs)

    run_perf_test(multinomial, [normalized_tensor_npu])
    print("----------- multinomial test success ------------")

if __name__ == "__main__":
    run_test()