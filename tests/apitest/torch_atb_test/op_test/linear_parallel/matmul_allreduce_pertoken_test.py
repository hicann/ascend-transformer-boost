import torch
import torch_atb
import torch_npu
import torch.multiprocessing as mp
import re
import unittest
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

rank_size = 2
rank_root = 0
backend = "lcoc"
random_seed = 123

def create_op_run(rank, tensor):
    linear_parallel_param = torch_atb.LinearParallelParam(
        trans_weight = False,
        rank = rank,
        rank_size = rank_size,
        rank_root = rank_root,
        backend = backend,
        type = torch_atb.LinearParallelParam.ParallelType.LINEAR_ALL_REDUCE,
        quant_type = torch_atb.LinearParallelParam.QuantType.QUANT_TYPE_PER_TOKEN
    )
    linear_parallel = torch_atb.Operation(linear_parallel_param)
    result = linear_parallel.forward(tensor)
    return result

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def run_test(rank, size):
    torch_npu.npu.set_device(rank)
    logging.info(f'Process {rank} started, using device npu:{rank}.')
    torch.manual_seed(random_seed)

    # m, k, n = 57600, 6144, 6144
    m, k, n = 2, 2, 2
    inTensors_npu = []
    for _ in range(rank_size):
        input = torch.randn(m, k, dtype=torch.float16).npu()
        weight = torch.randint(low=0, high=10, size=(k, n), dtype=torch.int8).npu()
        bias = torch.randn(2, dtype=torch.float16).npu()
        weight_scale = torch.randn(2, dtype=torch.float16).npu()
        input_scale = torch.randn(2, dtype=torch.float).npu()
        
        inTensors_npu.append([input, weight, bias, weight_scale, input_scale])

    npu_outputs = create_op_run(rank, inTensors_npu[rank])
    torch.npu.synchronize()
    npu_res_cur_rank = npu_outputs[0].cpu()
    logging.info(f"shape of npu_outputs for rank {rank}: {npu_res_cur_rank.shape}")
    logging.debug(f"npu_outputs for rank {rank}: {npu_res_cur_rank}")

    # assert check_precision(npu_res_cur_rank, cpu_res_cur_rank, rank), "Test failed"

class TestLinearParallel(unittest.TestCase):
    def test(self):
        if not is910B():
            print("This test case only supports 910B")
            return True
        print("----------- linear_parallel test begin ------------")
        mp.spawn(run_test, nprocs=rank_size, args=(rank_size,))
        print("----------- linear_parallel test success ------------")

if __name__ == "__main__":
    unittest.main()