import torch
import torch_atb
import torch_npu
import torch.multiprocessing as mp
import sys
import os
import re
import unittest
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

rank_size = 8
rank_root = 0
backend = "lcoc"
random_seed = 123

def check_precision(out_tensor, golden_out_tensor, rank, err = 2**-8):
    # 计算每个元素的误差阈值
    max_err = err * torch.max(torch.ones_like(golden_out_tensor), torch.abs(golden_out_tensor))

    # 计算实际误差
    error = torch.abs(out_tensor - golden_out_tensor)

    # 计算不满足条件的元素个数
    num_failures = torch.sum(error > max_err).item()
    if rank == 0:
        logging.info(f"num_failures: {num_failures}")
    return num_failures == 0

def compute_golden(inTensors, world_size):
    mats = [torch.matmul(inp.float(), wgh.float())
            for inp, wgh in inTensors]
    chunks_per_sender = [mat.chunk(world_size, dim=1) for mat in mats]
    cpu_goldens = []
    for rank in range(world_size):
        pieces = [chunks_per_sender[sender][rank]
                  for sender in range(world_size)]
        golden = torch.cat(pieces, dim=0)
        cpu_goldens.append(golden)
    return cpu_goldens

def create_op_run(rank, tensor):
    linear_parallel_param = torch_atb.LinearParallelParam(
        trans_weight = False,
        rank = rank,
        rank_size = rank_size,
        rank_root = rank_root,
        backend = backend,
        type = torch_atb.LinearParallelParam.ParallelType.LINEAR_ALL_TO_ALL
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

    m, k, n = 2, 16, 32
    inTensors = []
    inTensors_npu = []
    for _ in range(rank_size):
        input_tensor = torch.randn(m, k, dtype=torch.float16)
        weight_tensor = torch.randn(k, n, dtype=torch.float16)
        input_npu = input_tensor.npu()
        weight_npu = weight_tensor.npu()
        inTensors.append([input_tensor, weight_tensor])
        inTensors_npu.append([input_npu, weight_npu])

    cpu_goldens = compute_golden(inTensors, size)
    cpu_res_cur_rank = cpu_goldens[rank]
    logging.info(f"shape of cpu_goldens for rank {rank}: {cpu_res_cur_rank.shape}")
    logging.debug(f"cpu_goldens for rank {rank}: {cpu_res_cur_rank}")

    npu_outputs = create_op_run(rank, inTensors_npu[rank])
    torch.npu.synchronize()
    npu_res_cur_rank = npu_outputs[0].cpu()
    logging.info(f"shape of npu_outputs for rank {rank}: {npu_res_cur_rank.shape}")
    logging.debug(f"npu_outputs for rank {rank}: {npu_res_cur_rank}")

    assert check_precision(npu_res_cur_rank, cpu_res_cur_rank, rank), "Test failed"

if __name__ == "__main__":
    # if not is910B():
    #     print("This test case only supports 910B")
    #     return True
    print("----------- linear_parallel test begin ------------")
    mp.spawn(run_test, nprocs=rank_size, args=(rank_size,))
    print("----------- linear_parallel test success ------------")