import torch
import torch_npu
import torch_atb
import torch.multiprocessing as mp
import sys
import os
import re
import logging
import unittest
import logging

rank_root = 0
backend = "lcoc"
type=torch_atb.LinearParallelParam.ParallelType.LINEAR_ALL_TO_ALL

def linear_parallel_run(rank, size, tensors):
    linear_parallel_param = torch_atb.LinearParallelParam()
    linear_parallel_param.trans_weight = False
    linear_parallel_param.rank = rank
    linear_parallel_param.rank_size = size
    linear_parallel_param.rank_root = rank_root
    linear_parallel_param.backend = backend
    linear_parallel_param.type = type
    linear_parallel = torch_atb.Operation(linear_parallel_param)
    result = linear_parallel.forward(tensors)
    logging.info(linear_parallel_param)
    return result

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def run_test(rank, size):
    print("----------- linear_parallel test begin ------------")
    torch_npu.npu.set_device(rank)
    logging.info(f'Process {rank} started, using device npu:{rank}.')

    m, k, n = 2, 16, 32
    input_tensor = torch.randn(m, k, dtype=torch.float16)
    weight_tensor = torch.randn(k, n, dtype=torch.float16)
    input = input_tensor.npu()
    weight = weight_tensor.npu()

    npu_outputs = linear_parallel_run(rank, size, [input, weight])
    torch.npu.synchronize()
    logging.info("npu_outputs: ", npu_outputs)

    linear_result = torch.matmul(input.to(torch.float32),weight.to(torch.float32))
    chunks =  torch.chunk(linear_result, size, dim=1)
    golden_result = [chunk.repeat(size,1) for chunk in chunks]
    golden_result = [golden_result[rank]]
    logging.info("cpu_outputs: ", golden_result)
    print("----------- linear test success ------------")

class TestGraph(unittest.TestCase):
    def test_1(self):
        if not is910B():
            print("This test case only supports 910B")
            return True
        print("----------- linear_parallel test begin ------------")
        rank_size=2
        mp.spawn(run_test, nprocs=rank_size, args=(rank_size,))
        print("----------- linear_parallel test begin ------------")

    def test_2(self):
        if not is910B():
            print("This test case only supports 910B")
            return True
        print("----------- linear_parallel test begin ------------")
        rank_size=4
        mp.spawn(run_test, nprocs=rank_size, args=(rank_size,))
        print("----------- linear_parallel test begin ------------")

    def test_3(self):
        if not is910B():
            print("This test case only supports 910B")
            return True
        print("----------- linear_parallel test begin ------------")
        rank_size=8
        mp.spawn(run_test, nprocs=rank_size, args=(rank_size,))
        print("----------- linear_parallel test begin ------------")
if __name__ == "__main__":
    unittest.main()